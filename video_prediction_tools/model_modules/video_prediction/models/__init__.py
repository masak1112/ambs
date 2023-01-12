# SPDX-FileCopyrightText: 2018, alexlee-gk
#
# SPDX-License-Identifier: MIT


from .vanilla_convLSTM_model import VanillaConvLstmVideoPredictionModel
from .test_model import TestModelVideoPredictionModel
from model_modules.model_architectures import known_models
from .convLSTM_GAN_model import ConvLstmGANVideoPredictionModel


def get_model_class(model):
    model_mappings = known_models()
    model_class = model_mappings.get(model, model)
    model_class = globals().get(model_class)
    if model_class is None:
        raise ValueError('Invalid model %s' % model)
    return model_class
