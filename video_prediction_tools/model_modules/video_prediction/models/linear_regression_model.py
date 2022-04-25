# SPDX-FileCopyrightText: 2018, alexlee-gk
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-04-13"

from .our_base_model import BaseModels
import tensorflow as tf


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
            self.opt_var = self.hparams.opt_var
            self.learning_rate = self.hparams.lr

            print("The model hparams have been parsed successfully! ")
        except Exception as error:
           print("Method %{}: error: {}".format(method, error))
           raise("Method %{}: the hparameter dictionary must include the params defined above!".format(method))

    def build_graph(self, x: tf.Tensor):

        self.is_build_graph = False
        self.inputs = x
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()

        self.build_model()


        # This is the loss function (MSE):
        # Optimize all target variables/channels
        if self.opt_var == "all":
            x = self.inputs[:, self.context_frames:, :, :, :]
            x_hat = self.x_hat_predict_frames[:, :, :, :, :]
            print("The model is optimzied on all the variables in the loss function")
        elif self.opt_var != "all" and isinstance(self.opt_var, str):
            self.opt_var = int(self.opt_var)
            print("The model is optimized on the {} variable in the loss function".format(self.opt_var))
            x = self.inputs[:, self.context_frames:, :, :, self.opt_var]
            x_hat = self.x_hat_predict_frames[:, :, :, :, self.opt_var]
        else:
            raise ValueError(
                "The opt var in the hyper-parameters setup should be '0','1','2' indicate the index of target variable to be optimised or 'all' indicating optimize all the variables")

        #loss function is mean squre error
        self.total_loss = tf.reduce_mean(tf.square(x - x_hat))

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
        pass
