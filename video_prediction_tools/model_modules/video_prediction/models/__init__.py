from .base_model import BaseVideoPredictionModel
from .base_model import VideoPredictionModel
from .non_trainable_model import NonTrainableVideoPredictionModel
from .non_trainable_model import GroundTruthVideoPredictionModel
from .non_trainable_model import RepeatVideoPredictionModel
from .savp_model import SAVPVideoPredictionModel
from .dna_model import DNAVideoPredictionModel
from .sna_model import SNAVideoPredictionModel
from .sv2p_model import SV2PVideoPredictionModel
from .vanilla_vae_model import VanillaVAEVideoPredictionModel
from .vanilla_convLSTM_model import VanillaConvLstmVideoPredictionModel
from .mcnet_model import McNetVideoPredictionModel
from model_modules.model_architectures import known_models

def get_model_class(model):
    model_mappings = known_models()
    model_class = model_mappings.get(model, model)
    model_class = globals().get(model_class)
    if model_class is None or not issubclass(model_class, BaseVideoPredictionModel):
        raise ValueError('Invalid model %s' % model)
    return model_class
