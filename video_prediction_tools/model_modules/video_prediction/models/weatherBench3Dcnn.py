# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT
# Weather Bench models
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2021-04-13"

import tensorflow as tf
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.losses import l1_loss
from .our_base_model import BaseModels

class WeatherBenchModel(BaseModels):

    filters = [64, 64, 64, 64, 2]
    kernels = [5, 5, 5, 5, 5]

    def __init__(self, hparams_dict_config: dict=None, mode:str="train", **kwargs):
        """
        This is class for building weatherBench architecture by using updated hparameters
        args:
             mode         :"train" or "val", side note: mode may not be used in the convLSTM,
                          this will be a useful argument for the GAN-based model
             hparams_dict :The dictionary contains the hyper-parameters names and values
        """
        super().__init__(hparams_dict_config, mode)


    def parse_hparams(self, hparams):
        """
        Obtain the hyper-parameters from the dict to the class variables
        """
        try:
            self.context_frames = self.hparams.context_frames
            self.max_epochs = self.hparams.max_epochs
            self.batch_size = self.hparams.batch_size
            self.shuffle_on_val = self.hparams.shuffle_on_val
            self.loss_fun = self.hparams.loss_fun
            self.learning_rate = self.hparams.lr
            self.sequence_length = self.hparams.sequence_length
        except Exception as error:
           print("error: {}".format(error))
           raise ValueError("Method %{}: the hyper-parameter dictionary "
                            "must include parameters above")

    def get_loss(self, x: tf.Tensor, x_hat: tf.Tensor):
        # Loss
        total_loss = l1_loss(x[:, 1, :, :, :], x_hat[:, :, :, :])
        return total_loss

    def optimizer(self, total_loss):
        return tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(total_loss,
                                                         global_step =
                                                         self.global_step)

    def build_model(self, x):
        """Fully convolutional network"""
        x = x[:, 0, :, :, :]

        _idx = 0
        for f, k in zip(filters[:-1], kernels[:-1]):
            with tf.variable_scope("conv_layer_"+str(_idx), reuse=tf.AUTO_REUSE):
                x = ld.conv_layer(x, kernel_size=k, stride=1,
                                  num_features=f,
                                  idx="conv_layer_"+str(_idx),
                                  activate="leaky_relu")
            _idx += 1
        with tf.variable_scope("conv_last_layer", reuse=tf.AUTO_REUSE):
            output = ld.conv_layer(x, kernel_size=kernels[-1],
                                   stride=1, num_features=filters[-1],
                                   idx="conv_last_layer", activate="linear")

        return output


    def forecast(self, x, forecast_time):
        x_hat = []

        for i in range(forecast_time):
            if i == 0:
                x_pred = self.build_model(x[:, i, :, :, :], filters, kernels)
            else:
                x_pred = self.build_model(_x_pred, _filters, kernels)
            x_hat.append(x_pred)

        x_hat = tf.stack(x_hat)
        x_hat = tf.transpose(x_hat, [1, 0, 2, 3, 4])
        return x_hat


