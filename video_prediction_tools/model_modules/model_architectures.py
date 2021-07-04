def known_models():
    """
    An auxilary function
    ours_vae_l1 and ours_gan are from savp papers
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
        'convLSTM_gan': "ConvLstmGANVideoPredictionModel",
        'ours_vae_l1': 'SAVPVideoPredictionModel',
        'ours_gan': 'SAVPVideoPredictionModel',
        'precrnn_v2': 'PredRNNv2VideoPredictionModel'
        }

    return model_mappings
