def known_datasets():
    """
    An auxilary function
    :return: dictionary of known datasets
    """
    dataset_mappings = {
        'google_robot': 'GoogleRobotVideoDataset',
        'sv2p': 'SV2PVideoDataset',
        'softmotion': 'SoftmotionVideoDataset',
        'bair': 'SoftmotionVideoDataset',  # alias of softmotion
        'kth': 'KTHVideoDataset',
        'ucf101': 'UCF101VideoDataset',
        'cartgripper': 'CartgripperVideoDataset',
        "era5": "ERA5Dataset",
        "moving_mnist": "MovingMnist"
        #        "era5_anomaly":"ERA5Dataset_v2_anomaly",
    }

    return dataset_mappings
