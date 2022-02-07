def known_models():
    """
    An auxilary function
    ours_vae_l1 and ours_gan are from savp papers
    :return: dictionary of known model architectures
    """
    model_mappings = {
        'ground_truth': 'GroundTruthVideoPredictionModel',
        'savp': 'SAVPVideoPredictionModel',
        'vae': 'VanillaVAEVideoPredictionModel',
        'convLSTM': 'VanillaConvLstmVideoPredictionModel',
        'mcnet': 'McNetVideoPredictionModel',
        'convLSTM_gan': "ConvLstmGANVideoPredictionModel",
        'ours_vae_l1': 'SAVPVideoPredictionModel',
        'ours_gan': 'SAVPVideoPredictionModel',
    }

    return model_mappings
