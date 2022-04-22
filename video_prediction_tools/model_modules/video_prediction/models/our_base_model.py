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
        self.hparams_dict_config = hparams_dict_config
        self.hparams_dict = self.get_model_hparams_dict()
        self.hparams = self.parse_hparams()
        # Attributes set during runtime
        self.total_loss = None
        self.loss_summary = None
        self.total_loss = None
        self.outputs = {}
        self.train_op = None
        self.summary_op = None
        self.inputs = None
        self.global_step = None
        self.saveable_variables = None
        self.is_build_graph = None
        self.x_hat = None
        self.x_hat_predict_frames = None


    def get_model_hparams_dict(self):
        """
        Get model_hparams_dict from json file
        """
        if self.hparams_dict_config:
            with open(self.hparams_dict_config, 'r') as f:
                hparams_dict = json.loads(f.read())
        else:
            raise FileNotFoundError("hparam directory doesn't exist! please check {}!".format(self.hparams_dict_config))

        return hparams_dict

    def parse_hparams(self):
        """
        Obtain the parameters from directory
        """

        hparams = dotdict(self.hparams_dict)
        return hparams

    @abstractmethod
    def get_hparams(self):
        """
        obtain the hparams from the dict to the class variables
        """
        method = BaseModels.get_hparams.__name__

        try:
            self.context_frames = self.hparams.context_frames
            self.max_epochs = self.hparams.max_epochs
            self.batch_size = self.hparams.batch_size
            self.shuffle_on_val = self.hparams.shuffle_on_val
            self.loss_fun = self.hparams.loss_fun

        except Exception as error:
           print("Method %{}: error: {}".format(method,error))
           raise("Method %{}: the hparameter dictionary must include "
                 "'context_frames','max_epochs','batch_size','shuffle_on_val' 'loss_fun'".format(method))



    @abstractmethod
    def build_graph(self, x: tf.Tensor):

        pass


    @abstractmethod
    def build_model(self):
        pass
