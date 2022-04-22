# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"
__date__ = "2020-11-05"

from model_modules.video_prediction.models.model_helpers import set_and_check_pred_frames
import tensorflow as tf
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.layers.BasicConvLSTMCell import BasicConvLSTMCell
from .our_base_model import BaseModels

class VanillaConvLstmVideoPredictionModel(BaseModels):
    def __init__(self, hparams_dict=None, **kwargs):
        """
        This is class for building convLSTM architecture by using updated hparameters
        args:
             hparams_dict : dict, the dictionary contains the hparaemters names and values
        """
        super().__init__(hparams_dict)
        self.get_hparams()


    def get_hparams(self):
        """
        obtain the hparams from the dict to the class variables
        """
        method = BaseModels.get_hparams.__name__

        try:
            self.context_frames = self.hparams.context_frames
            self.sequence_length = self.hparams.sequence_length
            self.max_epochs = self.hparams.max_epochs
            self.batch_size = self.hparams.batch_size
            self.shuffle_on_val = self.hparams.shuffle_on_val
            self.loss_fun = self.hparams.loss_fun
            self.opt_var = self.hparams.opt_var
            self.learning_rate = self.hparams.lr
            self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
            print("The model hparams have been parsed successfully! ")
        except Exception as error:
           print("Method %{}: error: {}".format(method, error))
           raise("Method %{}: the hparameter dictionary must include "
                 "'context_frames','max_epochs','batch_size','shuffle_on_val' 'loss_fun',"
                 "'opt_var', 'lr', 'opt_var'".format(method))


    def build_graph(self, x):
        self.is_build_graph = False
        self.inputs = x
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()

        self.build_model()
        #This is the loss function (MSE):

        #Optimize all target variables/channels
        if self.opt_var == "all":
            x = self.inputs[:, self.context_frames:, :, :, :]
            x_hat = self.x_hat_predict_frames[:, :, :, :, :]
            print ("The model is optimzied on all the variables in the loss function")
        elif self.opt_var != "all" and isinstance(self.opt_var, str):
            self.opt_var = int(self.opt_var)
            print ("The model is optimized on the {} variable in the loss function".format(self.opt_var))
            x = self.inputs[:, self.context_frames:, :, :, self.opt_var]
            x_hat = self.x_hat_predict_frames[:, :, :, :, self.opt_var]
        else:
            raise ValueError("The opt var in the hyperparameters setup should be '0','1','2' indicate the index of target variable to be optimised or 'all' indicating optimize all the variables")

        if self.loss_fun == "mse":
            self.total_loss = tf.reduce_mean(tf.square(x - x_hat))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(x, [-1])
            x_hat_predict_frames_flatten = tf.reshape(x_hat, [-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            self.total_loss = bce(x_flatten, x_hat_predict_frames_flatten)
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'mse' or 'cross_entropy'")

        #This is the loss for only all the channels(temperature, geo500, pressure)
        #self.total_loss = tf.reduce_mean(
        #    tf.square(self.x[:, self.context_frames:,:,:,:] - self.x_hat_predict_frames[:,:,:,:,:]))            
 
        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(self.total_loss, global_step = self.global_step)

        self.outputs["gen_images"] = self.x_hat
        # Summary op
        self.loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        self.is_build_graph = True
        return self.is_build_graph

    def build_model(self):
        network_template = tf.make_template('network',
                                            VanillaConvLstmVideoPredictionModel.convLSTM_cell)  # make the template to share the variables
        # create network
        x_hat = []
        
        #This is for training (optimization of convLSTM layer)
        hidden_g = None
        for i in range(self.sequence_length-1):
            if i < self.context_frames:
                x_1_g, hidden_g = network_template(self.inputs[:, i, :, :, :], hidden_g)
            else:
                x_1_g, hidden_g = network_template(x_1_g, hidden_g)
            x_hat.append(x_1_g)

        # pack them all together
        x_hat = tf.stack(x_hat)
        self.x_hat = tf.transpose(x_hat, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat_predict_frames = self.x_hat[:, self.context_frames-1:, :, :, :]

    @staticmethod
    def convLSTM_cell(inputs, hidden):
        y_0 = inputs #we only usd patch 1, but the original paper use patch 4 for the moving mnist case, but use 2 for Radar Echo Dataset
        channels = inputs.get_shape()[-1]
        # conv lstm cell
        cell_shape = y_0.get_shape().as_list()
        channels = cell_shape[-1]
        with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell(shape = [cell_shape[1], cell_shape[2]], filter_size=5, num_features=64)
            if hidden is None:
                hidden = cell.zero_state(y_0, tf.float32)
            output, hidden = cell(y_0, hidden)
        output_shape = output.get_shape().as_list()
        z3 = tf.reshape(output, [-1, output_shape[1], output_shape[2], output_shape[3]])
        #we feed the learn representation into a 1 × 1 convolutional layer to generate the final prediction
        x_hat = ld.conv_layer(z3, 1, 1, channels, "decode_1", activate="sigmoid")
        return x_hat, hidden

    @staticmethod
    def set_and_check_pred_frames(seq_length, context_frames):
        """
        Checks if sequence length and context_frames are set properly and returns number of frames to be predicted.
        :param seq_length: number of frames/images per sequences
        :param context_frames: number of context frames/images
        :return: number of predicted frames
        """

        method = VanillaConvLstmVideoPredictionModel.set_and_check_pred_frames.__name__

        # sanity checks
        assert isinstance(seq_length, int), "%{0}: Sequence length (seq_length) must be an integer".format(method)
        assert isinstance(context_frames, int), "%{0}: Number of context frames must be an integer".format(method)

        if seq_length > context_frames:
            return seq_length-context_frames
        else:
            raise ValueError("%{0}: Sequence length ({1}) must be larger than context frames ({2})."
                             .format(method, seq_length, context_frames))
