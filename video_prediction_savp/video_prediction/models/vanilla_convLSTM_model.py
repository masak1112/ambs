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
    def __init__(self, mode='train', hparams_dict=None,
                 hparams=None, **kwargs):
        super(VanillaConvLstmVideoPredictionModel, self).__init__(mode, hparams_dict, hparams, **kwargs)
        print ("Hparams_dict",self.hparams)
        self.mode = mode
        self.learning_rate = self.hparams.lr
        self.total_loss = None
        self.context_frames = self.hparams.context_frames
        self.sequence_length = self.hparams.sequence_length
        self.predict_frames = self.sequence_length - self.context_frames
        self.max_epochs = self.hparams.max_epochs
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
            max_steps: number of training steps.
            context_frames: the number of ground-truth frames to pass :qin at
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
            max_epochs=3000,
        )

        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def build_graph(self, x):
        self.x = x["images"]
        #self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        # ARCHITECTURE
        self.convLSTM_network()
        #print("self.x",self.x)
        #print("self.x_hat_context_frames,",self.x_hat_context_frames)
        #self.context_frames_loss = tf.reduce_mean(
        #    tf.square(self.x[:, :self.context_frames, :, :, 0] - self.x_hat_context_frames[:, :, :, :, 0]))
        # This is the loss function (RMSE):
        self.total_loss = tf.reduce_mean(
            tf.square(self.x[:, self.context_frames:, :, :, 0] - self.x_hat_context_frames[:, (self.context_frames-1):, :, :, 0]))

        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(self.total_loss, global_step = self.global_step)
        self.outputs = {}
        self.outputs["gen_images"] = self.x_hat
        # Summary op
        self.loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        return None


    @staticmethod
    def convLSTM_cell(inputs, hidden):
        channels = inputs.get_shape()[-1]
        conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1", activate = "leaky_relu")
        conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2", activate = "leaky_relu")
        conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3", activate = "leaky_relu")

        y_0 = conv3
        # conv lstm cell
        cell_shape = y_0.get_shape().as_list()
        with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell(shape = [cell_shape[1], cell_shape[2]], filter_size = [3, 3], num_features = 8)
            if hidden is None:
                hidden = cell.zero_state(y_0, tf.float32)

            output, hidden = cell(y_0, hidden)


        output_shape = output.get_shape().as_list()


        z3 = tf.reshape(output, [-1, output_shape[1], output_shape[2], output_shape[3]])

        conv5 = ld.transpose_conv_layer(z3, 3, 2, 8, "decode_5", activate = "leaky_relu")


        conv6 = ld.transpose_conv_layer(conv5, 3, 1, 8, "decode_6", activate = "leaky_relu")


        x_hat = ld.transpose_conv_layer(conv6, 3, 2, channels, "decode_7", activate = "sigmoid")  # set activation to linear

        return x_hat, hidden

    def convLSTM_network(self):
        network_template = tf.make_template('network',
                                            VanillaConvLstmVideoPredictionModel.convLSTM_cell)  # make the template to share the variables
        # create network
        x_hat_context = []
        x_hat = []
        hidden = None
        #This is for training 
        for i in range(self.sequence_length-1):
            if i < self.context_frames:
                x_1, hidden = network_template(self.x[:, i, :, :, :], hidden)
            else:
                x_1, hidden = network_template(x_1, hidden)
            x_hat_context.append(x_1)
        
        #This is for generating video
        hidden_g = None
        for i in range(self.sequence_length-1):
            if i < self.context_frames:
                x_1_g, hidden_g = network_template(self.x[:, i, :, :, :], hidden_g)
            else:
                x_1_g, hidden_g = network_template(x_1_g, hidden_g)
            x_hat.append(x_1_g)
        
        # pack them all together
        x_hat_context = tf.stack(x_hat_context)
        x_hat = tf.stack(x_hat)
        self.x_hat_context_frames = tf.transpose(x_hat_context, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat= tf.transpose(x_hat, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat_predict_frames = self.x_hat[:,self.context_frames:,:,:,:]
