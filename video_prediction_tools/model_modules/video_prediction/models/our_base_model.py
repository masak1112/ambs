# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-04-13"

from hparams_utils import *
import json
from abc import ABC, abstractmethod
import tensorflow as tf


class BaseModels(ABC):

    def __init__(self, hparams_dict_config=None):
        self.__model = None
        #self.__hparams = {}
        #Get the user defined hyper-parameters
        #if hparams_dict_config:
        #    with open(hparams_dict_config, 'r') as f:
        #        hparams_dict = json.loads(f.read())
        #else:
        #    raise FileNotFoundError("hyper-parameter directory doesn't exist! please check {}!".format(hparams_dict_config))

        self.__hparams = self.hparams_options(hparams_dict_config)
        self.parse_hparams(self.__hparams)

        # Compile options, must be custeromised in the sub-class
        self.inputs = None
        self.train_op = None
        self.total_loss = None
        self.outputs = {}
        self.loss_summary = None
        self.summary_op = None
        self.global_step = tf.train.get_or_create_global_step()
        self.saveable_variables = None
        self._is_build_graph_set = False

    
    def hparams_options(self, hparams_dict_config:str):
        if hparams_dict_config:
            with open(hparams_dict_config, 'r') as f:
                hparams_dict = json.loads(f.read())
        else:
            raise FileNotFoundError("hyper-parameter directory doesn't exist! please check {}!".format(hparams_dict_config))
        return dotdict(hparams_dict)


    @abstractmethod
    def parse_hparams(self, hparams)->None:
        """
        parse the hyper-parameter as class attribute
        Examples:
            ... code-block:: python
            def parse_hparams(self):
                try:
                    self.context_frames = hparams.context_frames
                    self.max_epochs = hparams.max_epochs
                    self.batch_size = hparams.batch_size
                    self.shuffle_on_val = hparams.shuffle_on_val
                    self.loss_fun = hparams.loss_fun

                except Exception as e:
                    raise ValueError(f"missing hyperparameter: {e.args[0]}")
        """
        pass


    def get_hparams(self):
        return self.__hparams


    @abstractmethod
    def build_graph(self, x: tf.Tensor)->bool:
        """
        This function is used for build the graph, and allow a optimiser to the graph by using tensorflow function.

        Example:
            ... code-block:: python
                def build_graph(self, inputs):
                    original_global_variables = tf.global_variables()
                    x_hat = self.build_model(x)
                    self.train_loss = self.get_loss(x,x_hat)
                    self.train_op = self.optimizer(self.train_loss)
                    self.outputs["gen_images"] = x_hat
                    self.summary_op = self.summary() #This is optional
                    global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
                    self.saveable_variables = [self.global_step] + global_variables
                    self._is_build_graph_set=True
                    return self._is_build_graph_set

        """
        pass

    @abstractmethod
    def optimizer(self, train_loss):
        """
        Define the optimizer
        Example:
            ... code-block:: python
                def optimizer(self):
                    train_op = tf.train.AdamOptimizer(
                        learning_rate = self.lr).minimize(total_loss, global_step = self.global_step)
                    return train_op
        """

        pass


    @abstractmethod
    def get_loss(self, x:tf.Tensor, x_hat:tf.Tensor)->tf.Tensor:
        """
        :param x    : Input tensors
        :param x_hat: Prediction/output tensors
        :return     : the loss function
        """
        pass


    @abstractmethod
    def summary(self,**kwargs):
        """
        return the summary operation can be used for TensorBoard
        """
        #self.loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        #self.summary_op = tf.summary.merge_all()
        pass
    @abstractmethod
    def build_model(self, x)->tf.Tensor:
        """
        This function is used to create the network
        Example: see example in vanilla_convLSTM_model.py, it must return prediction fnsrames and save it to the self.output
        which is used for calculating the loss
        """
        self.inputs = x
        pass







