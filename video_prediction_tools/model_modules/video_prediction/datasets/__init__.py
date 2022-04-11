#from .base_dataset import BaseVideoDataset
from .era5_dataset import ERA5Dataset
from .gzprcp_dataset import GzprcpDataset
from .moving_mnist import MovingMnist
from data_preprocess.dataset_options import known_datasets

def get_dataset_class(dataset):
    dataset_mappings = known_datasets()
    dataset_class = dataset_mappings.get(dataset, dataset)
    print("datset_class",dataset_class)
    if dataset_class is None:
        raise ValueError('Invalid dataset %s' % dataset)
    else:
        dataset_class = globals().get(dataset_class)
    return dataset_class
