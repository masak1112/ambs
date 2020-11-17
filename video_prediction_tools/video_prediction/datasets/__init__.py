from .base_dataset import BaseVideoDataset
from .base_dataset import VideoDataset, SequenceExampleVideoDataset, VarLenFeatureVideoDataset
from .google_robot_dataset import GoogleRobotVideoDataset
from .sv2p_dataset import SV2PVideoDataset
from .softmotion_dataset import SoftmotionVideoDataset
from .kth_dataset import KTHVideoDataset
from .ucf101_dataset import UCF101VideoDataset
from .cartgripper_dataset import CartgripperVideoDataset
from .era5_dataset import ERA5Dataset
from .moving_mnist import MovingMnist
#from .era5_dataset_v2_anomaly import ERA5Dataset_v2_anomaly

def get_dataset_class(dataset):
    dataset_mappings = {
        'google_robot': 'GoogleRobotVideoDataset',
        'sv2p': 'SV2PVideoDataset',
        'softmotion': 'SoftmotionVideoDataset',
        'bair': 'SoftmotionVideoDataset',  # alias of softmotion
        'kth': 'KTHVideoDataset',
        'ucf101': 'UCF101VideoDataset',
        'cartgripper': 'CartgripperVideoDataset',
        "era5":"ERA5Dataset",
        "moving_mnist":"MovingMnist"
#        "era5_anomaly":"ERA5Dataset_v2_anomaly",
    }
    dataset_class = dataset_mappings.get(dataset, dataset)
    print("datset_class",dataset_class)
    dataset_class = globals().get(dataset_class)
    if dataset_class is None:
        raise ValueError('Invalid dataset %s' % dataset)
    else:
        # ERA5Dataset does not inherit anything from VarLenFeatureVideoDataset-class, so it is the only dataset which
        # does not need to be a subclass of BaseVideoDataset
        if not dataset_class == "ERA5Dataset":
            if not issubclass(dataset_class,BaseVideoDataset):
                raise ValueError('Dataset {0} is not a valid dataset'.format(dataset_class))

    return dataset_class
