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
        self.loss_fun = self.hparams.loss_fun

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
            max_epochs: number of training epochs, each epoch equal to sample_size/batch_size
            loss_fun: string can be either "rmse" or "cross_entropy", loss function has to be set from the user 
         """
        default_hparams = super(VanillaConvLstmVideoPredictionModel, self).get_default_hparams_dict()
        print ("default hparams",default_hparams)
        hparams = dict(
            batch_size=16,
            lr=0.001,
            max_epochs=3000,
            loss_fun = None
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
        #This is loss function only for 1 channel (temperature RMSE)
        if self.loss_fun == "rmse":
            self.total_loss = tf.reduce_mean(
                tf.square(self.x[:, self.context_frames:,:,:,0] - self.x_hat_predict_frames[:,:,:,:,0]))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(self.x[:, self.context_frames:,:,:,0],[-1])
            x_hat_predict_frames_flatten = tf.reshape(self.x_hat_predict_frames[:,:,:,:,0],[-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            self.total_loss = bce(x_flatten,x_hat_predict_frames_flatten)  
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'rmse' or 'cross_entropy'")

        
        #This is the loss for only all the channels(temperature, geo500, pressure)
        #self.total_loss = tf.reduce_mean(
        #    tf.square(self.x[:, self.context_frames:,:,:,:] - self.x_hat_predict_frames[:,:,:,:,:]))            
 
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

        y_0 = inputs #we only usd patch 1, but the original paper use patch 4 for the moving mnist case, but use 2 for Radar Echo Dataset
        channels = inputs.get_shape()[-1]
        # conv lstm cell
        cell_shape = y_0.get_shape().as_list()
        channels = cell_shape[-1]
        with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell(shape = [cell_shape[1], cell_shape[2]], filter_size = [5, 5], num_features = 256)
            if hidden is None:
                hidden = cell.zero_state(y_0, tf.float32)
            output, hidden = cell(y_0, hidden)
        output_shape = output.get_shape().as_list()
        z3 = tf.reshape(output, [-1, output_shape[1], output_shape[2], output_shape[3]])
        #we feed the learn representation into a 1 × 1 convolutional layer to generate the final prediction
        x_hat = ld.conv_layer(z3, 1, 1, channels, "decode_1", activate="sigmoid")

        return x_hat, hidden

    def convLSTM_network(self):
        network_template = tf.make_template('network',
                                            VanillaConvLstmVideoPredictionModel.convLSTM_cell)  # make the template to share the variables
        # create network
        x_hat = []

        # for i in range(self.sequence_length-1):
        #     if i < self.context_frames:
        #         x_1, hidden = network_template(self.x[:, i, :, :, :], hidden)
        #     else:
        #         x_1, hidden = network_template(x_1, hidden)
        #     x_hat_context.append(x_1)
        
        #This is for training (optimization of convLSTM layer)
        hidden_g = None
        for i in range(self.sequence_length-1):
            if i < self.context_frames:
                x_1_g, hidden_g = network_template(self.x[:, i, :, :, :], hidden_g)
            else:
                x_1_g, hidden_g = network_template(x_1_g, hidden_g)
            x_hat.append(x_1_g)
        
        # pack them all together
        x_hat = tf.stack(x_hat)
        self.x_hat= tf.transpose(x_hat, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat_predict_frames = self.x_hat[:,self.context_frames-1:,:,:,:]
