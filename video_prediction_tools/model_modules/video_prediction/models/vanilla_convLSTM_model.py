# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-11-05"

from model_modules.video_prediction.models.model_helpers import set_and_check_pred_frames
import tensorflow as tf
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.layers.BasicConvLSTMCell import BasicConvLSTMCell
from .our_base_model import BaseModels
from hparams_utils import *

class VanillaConvLstmVideoPredictionModel(BaseModels):
    def __init__(self, hparams_dict_config=None, **kwargs):
        """
        This is class for building convLSTM architecture by using updated hparameters
        args:
            hparams_dict : dict, the dictionary contains the hparaemters names and values
        """
        super().__init__(hparams_dict_config)
        self.hparams = self.hparams_options(hparams_dict_config)
        self.parse_hparams(self.hparams)

    def parse_hparams(self, hparams):
        """
        obtain the hyper-parameters from the dictionary
        """

        try:
            self.context_frames = hparams.context_frames
            self.sequence_length = hparams.sequence_length
            self.max_epochs = hparams.max_epochs
            self.batch_size = hparams.batch_size
            self.shuffle_on_val = hparams.shuffle_on_val
            self.loss_fun = hparams.loss_fun
            self.opt_var = hparams.opt_var
            self.lr = hparams.lr
            self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
            print("The model hparams have been parsed successfully! ")
        except Exception as e:
            raise ValueError(f"missing hyperparameter: {e.args[0]}")



    def build_graph(self, x:tf.Tensor):
        self.inputs = x
        original_global_variables = tf.global_variables()
        x_hat = self.build_model(x)
        self.total_loss = self.get_loss(x, x_hat)
        self.train_op = self.optimizer(self.total_loss)
        self.outputs["gen_images"] = x_hat
        self.summary_op = self.summary(total_loss = self.total_loss)
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        self._is_build_graph_set = True
        return self._is_build_graph_set



    def get_loss(self, x:tf.Tensor, x_hat:tf.Tensor)->tf.Tensor:
        """
        :param x    : Input tensors
        :param x_hat: Prediction/output tensors
        :return     : the loss function
        """
        #This is the loss function (MSE):
        #Optimize all target variables/channels
        if self.opt_var == "all":
            x = x[:, self.context_frames:, :, :, :]
            print("The model is optimzied on all the variables in the loss function")
        elif self.opt_var != "all" and isinstance(self.opt_var, str):
            self.opt_var = int(self.opt_var)
            print("The model is optimized on the {} variable in the loss function".format(self.opt_var))
            x = x[:, self.context_frames:, :, :, self.opt_var]
            x_hat = x_hat[:, :, :, :, self.opt_var]
        else:
            raise ValueError("The opt var in the hyperparameters setup should be '0','1','2' indicate the index of target variable to be optimised or 'all' indicating optimize all the variables")

        if self.loss_fun == "mse":
            total_loss = tf.reduce_mean(tf.square(x - x_hat))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(x, [-1])
            x_hat_predict_frames_flatten = tf.reshape(x_hat, [-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            total_loss = bce(x_flatten, x_hat_predict_frames_flatten)
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'mse' or 'cross_entropy'")
        return total_loss



    def summary(self, total_loss)->None:
        """
        return the summary operation can be used for TensorBoard
        """
        tf.summary.scalar("total_loss", total_loss)
        summary_op = tf.summary.merge_all()
        return summary_op

    def build_model(self, x):
        network_template = tf.make_template('network',
                                            VanillaConvLstmVideoPredictionModel.convLSTM_cell)  # make the template to share the variables
        # create network
        x_hat = []
        
        #This is for training (optimization of convLSTM layer)
        hidden_g = None
        for i in range(self.sequence_length-1):
            if i < self.context_frames:
                x_1_g, hidden_g = network_template(x[:, i, :, :, :], hidden_g)
            else:
                x_1_g, hidden_g = network_template(x_1_g, hidden_g)
            x_hat.append(x_1_g)

        # pack them all together
        x_hat = tf.stack(x_hat)
        x_hat = tf.transpose(x_hat, [1, 0, 2, 3, 4])  # change first dim with sec dim
        x_hat = x_hat[:, self.context_frames-1:, :, :, :]
        return x_hat

    @staticmethod
    def convLSTM_cell(inputs, hidden):
        y_0 = inputs #we only usd patch 1, but the original paper use patch 4 for the moving mnist case, but use 2 for Radar Echo Dataset

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
