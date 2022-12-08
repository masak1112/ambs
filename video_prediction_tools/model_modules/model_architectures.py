def known_models():
    """
    An auxilary function
    ours_vae_l1 and ours_gan are from savp papers
    :return: dictionary of known model architectures
    """
    model_mappings = {
        'ground_truth': 'GroundTruthVideoPredictionModel',
        'savp': 'SAVPVideoPredictionModel',
        'convLSTM': 'VanillaConvLstmVideoPredictionModel',
        'convLSTM_gan': "ConvLstmGANVideoPredictionModel",
        'weatherBench': 'WeatherBenchModel'
        }

    return model_mappings
