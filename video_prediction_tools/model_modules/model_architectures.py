def known_models():
    """
    An auxilary function
    :return: dictionary of known model architectures
    """
    model_mappings = {
        'convLSTM': 'VanillaConvLstmVideoPredictionModel',
        'convLSTM_gan': "ConvLstmGANVideoPredictionModel",
        'weatherBench': 'WeatherBenchModel'
        }

    return model_mappings
