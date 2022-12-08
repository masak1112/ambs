# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT
# Weather Bench models
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2021-04-13"

import tensorflow as tf
from tensorflow.contrib.training import HParams
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.losses import  *

class WeatherBenchModel(object):

    def __init__(self, hparams_dict=None, mode="train",**kwargs):
        """
        This is class for building weahterBench architecture by using updated hparameters
        args:
             mode        :str, "train" or "val", side note: mode may not be used in the convLSTM, but this will be a useful argument for the GAN-based model
             hparams_dict: dict, the dictionary contains the hparaemters names and values
        """
        self.hparams_dict = hparams_dict
        self.mode = mode
        self.hparams = self.parse_hparams()
        self.learning_rate = self.hparams.lr
        self.filters = self.hparams.filters
        self.kernels = self.hparams.kernels
        self.max_epochs = self.hparams.max_epochs
        self.batch_size = self.hparams.batch_size
        self.outputs = {}
        self.total_loss = None

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self):
        """
        Parse the hparams setting to ovoerride the default ones
        """

        parsed_hparams = self.get_default_hparams().override_from_dict(self.hparams_dict or {})
        return parsed_hparams


    def get_default_hparams_dict(self):
        """
        The function that contains default hparams
        Returns:
            A dict with the following hyperparameters.
            context_frames  : the number of ground-truth frames to pass in at start.
            max_epochs      : the number of epochs to train model
            lr              : learning rate
            loss_fun        : the loss function
            filters         : list contains the filters of each convolutional layer
            kernels         : list contains the kernels size for each convolutional layer
            """
        hparams = dict(
            sequence_length =13,
            context_frames =1,
            max_epochs = 20,
            batch_size = 40,
            lr = 0.001,
            shuffle_on_val= True,
            filters = [64, 64, 64, 64, 3],
            kernels = [5, 5, 5, 5, 5]
        )
        return hparams


    def build_graph(self, x):
        self.is_build_graph = False
        self.x = x["images"]

        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()

        # Architecture
        x_hat = self.build_model(self.x[:,0,:, :, :],self.filters, self.kernels)
        # Loss
        
        self.total_loss = l1_loss(self.x[:,1,:, :,:], x_hat[:,:,:,:])

        # Optimizer
        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(self.total_loss, global_step = self.global_step)

        # outputs
        self.outputs["total_loss"] = self.total_loss
       
        # inferences
        if self.mode == "test":
            self.outputs["gen_images"] = self.forecast(self.x, 12, self.filters, self.kernels)
        else:
            self.outputs["gen_images"] = x_hat

        # Summary op
        tf.summary.scalar("total_loss", self.total_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        self.is_build_graph = True
        return self.is_build_graph


    def build_model(self, x, filters, kernels):
        """Fully convolutional network"""
        idx = 0 
        for f, k in zip(filters[:-1], kernels[:-1]):
            with tf.variable_scope("conv_layer_"+str(idx),reuse=tf.AUTO_REUSE):
                x = ld.conv_layer(x, kernel_size=k, stride=1, num_features=f, idx="conv_layer_"+str(idx) , activate="leaky_relu")
            idx += 1
        with tf.variable_scope("Conv_last_layer",reuse=tf.AUTO_REUSE):
            output = ld.conv_layer(x, kernel_size=kernels[-1], stride=1, num_features=filters[-1], idx="Conv_last_layer", activate="linear")
        return output


    def forecast(self, x, forecast_time, filters, kernels):
        x_hat = []

        for i in range(forecast_time):
            if i == 0:
                x_pred = self.build_model(x[:,i,:, :,:],filters,kernels)
            else:
                x_pred = self.build_model(x_pred,filters,kernels)
            x_hat.append(x_pred)

        x_hat = tf.stack(x_hat)
        x_hat = tf.transpose(x_hat, [1, 0, 2, 3, 4])
        return x_hat


