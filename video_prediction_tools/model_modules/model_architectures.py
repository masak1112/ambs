def known_models():
    """
    An auxilary function
    :return: dictionary of known model architectures
    """
    model_mappings = {
        'ground_truth': 'GroundTruthVideoPredictionModel',
        'repeat': 'RepeatVideoPredictionModel',
        'savp': 'SAVPVideoPredictionModel',
        'dna': 'DNAVideoPredictionModel',
        'sna': 'SNAVideoPredictionModel',
        'sv2p': 'SV2PVideoPredictionModel',
        'vae': 'VanillaVAEVideoPredictionModel',
        'convLSTM': 'VanillaConvLstmVideoPredictionModel',
        'mcnet': 'McNetVideoPredictionModel',
        'gan': "VanillaGANVideoPredictionModel",
         "convLSTM_gan": "ConvLstmGANVideoPredictionModel"
    }

    return model_mappings
