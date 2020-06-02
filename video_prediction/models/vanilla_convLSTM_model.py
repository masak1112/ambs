import collections
import functools
import itertools
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from video_prediction import ops, flow_ops
from video_prediction.models import BaseVideoPredictionModel
from video_prediction.models import networks
from video_prediction.ops import dense, pad2d, conv2d, flatten, tile_concat
from video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from video_prediction.utils import tf_utils
from datetime import datetime
from pathlib import Path
from video_prediction.layers import layer_def as ld
from video_prediction.layers.BasicConvLSTMCell import BasicConvLSTMCell

class VanillaConvLstmVideoPredictionModel(BaseVideoPredictionModel):
    def __init__(self, mode='train',aggregate_nccl=None, hparams_dict=None,
                 hparams=None, **kwargs):
        super(VanillaConvLstmVideoPredictionModel, self).__init__(mode, hparams_dict, hparams, **kwargs)
        print ("Hparams_dict",self.hparams)
        self.mode = mode
        self.learning_rate = self.hparams.lr
        self.gen_images_enc = None
        self.recon_loss = None
        self.latent_loss = None
        self.total_loss = None
        self.context_frames = 10
        self.sequence_length = 20
        self.predict_frames = self.sequence_length - self.context_frames
        self.aggregate_nccl=aggregate_nccl
    
    def get_default_hparams_dict(self):
        """
        The keys of this dict define valid hyperparameters for instances of
        this class. A class inheriting from this one should override this
        method if it has a different set of hyperparameters.

        Returns:
            A dict with the following hyperparameters.

            batch_size: batch size for training.
            lr: learning rate. if decay steps is non-zero, this is the
                learning rate for steps <= decay_step.
            end_lr: learning rate for steps >= end_decay_step if decay_steps
                is non-zero, ignored otherwise.
            decay_steps: (decay_step, end_decay_step) tuple.
            max_steps: number of training steps.
            beta1: momentum term of Adam.
            beta2: momentum term of Adam.
            context_frames: the number of ground-truth frames to pass in at
                start. Must be specified during instantiation.
            sequence_length: the number of frames in the video sequence,
                including the context frames, so this model predicts
                `sequence_length - context_frames` future frames. Must be
                specified during instantiation.
        """
        default_hparams = super(VanillaConvLstmVideoPredictionModel, self).get_default_hparams_dict()
        print ("default hparams",default_hparams)
        hparams = dict(
            batch_size=16,
            lr=0.001,
            end_lr=0.0,
            nz=16,
            decay_steps=(200000, 300000),
            max_steps=350000,
        )

        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def build_graph(self, x):
        self.x = x["images"]
        #self.global_step = tf.train.get_or_create_global_step()
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        original_global_variables = tf.global_variables()
        # ARCHITECTURE
        self.x_hat_context_frames, self.x_hat_predict_frames = self.convLSTM_network()
        self.x_hat = tf.concat([self.x_hat_context_frames, self.x_hat_predict_frames], 1)
        print("x_hat,shape", self.x_hat)

        self.context_frames_loss = tf.reduce_mean(
            tf.square(self.x[:, :self.context_frames, :, :, 0] - self.x_hat_context_frames[:, :, :, :, 0]))
        self.predict_frames_loss = tf.reduce_mean(
            tf.square(self.x[:, self.context_frames:, :, :, 0] - self.x_hat_predict_frames[:, :, :, :, 0]))
        self.total_loss = self.context_frames_loss + self.predict_frames_loss

        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(self.total_loss, global_step = self.global_step)
        self.outputs = {}
        self.outputs["gen_images"] = self.x_hat
        # Summary op
        self.loss_summary = tf.summary.scalar("recon_loss", self.context_frames_loss)
        self.loss_summary = tf.summary.scalar("latent_loss", self.predict_frames_loss)
        self.loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        return


    @staticmethod
    def convLSTM_cell(inputs, hidden, nz=16):
        print("Inputs shape", inputs.shape)
        conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1", activate = "leaky_relu")
        print("Encode_1_shape", conv1.shape)
        conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2", activate = "leaky_relu")
        print("Encode 2_shape,", conv2.shape)
        conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3", activate = "leaky_relu")
        print("Encode 3_shape, ", conv3.shape)
        y_0 = conv3
        # conv lstm cell
        cell_shape = y_0.get_shape().as_list()
        with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell(shape = [cell_shape[1], cell_shape[2]], filter_size = [3, 3], num_features = 8)
            if hidden is None:
                hidden = cell.zero_state(y_0, tf.float32)
                print("hidden zero layer", hidden.shape)
            output, hidden = cell(y_0, hidden)
            print("output for cell:", output)

        output_shape = output.get_shape().as_list()
        print("output_shape,", output_shape)

        z3 = tf.reshape(output, [-1, output_shape[1], output_shape[2], output_shape[3]])

        conv5 = ld.transpose_conv_layer(z3, 3, 2, 8, "decode_5", activate = "leaky_relu")
        print("conv5 shape", conv5)

        conv6 = ld.transpose_conv_layer(conv5, 3, 1, 8, "decode_6", activate = "leaky_relu")
        print("conv6 shape", conv6)

        x_hat = ld.transpose_conv_layer(conv6, 3, 2, 3, "decode_7", activate = "sigmoid")  # set activation to linear
        print("x hat shape", x_hat)
        return x_hat, hidden

    def convLSTM_network(self):
        network_template = tf.make_template('network',
                                            VanillaConvLstmVideoPredictionModel.convLSTM_cell)  # make the template to share the variables
        # create network
        x_hat_context = []
        x_hat_predict = []
        seq_start = 1
        hidden = None
        for i in range(self.context_frames):
            if i < seq_start:
                x_1, hidden = network_template(self.x[:, i, :, :, :], hidden)
            else:
                x_1, hidden = network_template(x_1, hidden)
            x_hat_context.append(x_1)

        for i in range(self.predict_frames):
            x_1, hidden = network_template(x_1, hidden)
            x_hat_predict.append(x_1)

        # pack them all together
        x_hat_context = tf.stack(x_hat_context)
        x_hat_predict = tf.stack(x_hat_predict)
        self.x_hat_context = tf.transpose(x_hat_context, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat_predict = tf.transpose(x_hat_predict, [1, 0, 2, 3, 4])  # change first dim with sec dim
        return self.x_hat_context, self.x_hat_predict
