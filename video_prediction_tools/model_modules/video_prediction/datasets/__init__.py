#from .base_dataset import BaseVideoDataset
from .era5_dataset import ERA5Dataset
from .gzprcp_dataset import GzprcpDataset
#from .moving_mnist import MovingMnist
from data_preprocess.dataset_options import known_datasets
from stats import MinMax, ZScore

known_datasets = ["era5", "weatherbench"]
default_normalization = {"era5": MinMax, "weatherbench": ZScore}


