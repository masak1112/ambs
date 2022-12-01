#from .base_dataset import BaseVideoDataset
#from .era5_dataset import ERA5Dataset
#from .gzprcp_dataset import GzprcpDataset
#from .moving_mnist import MovingMnist
#from data_preprocess.dataset_options import known_datasets
from .stats import MinMax, ZScore
from .dataset import Dataset
import dask
from dask.base import tokenize
from utils.dataset_utils import DATASETS, get_dataset_info, get_filename_template


def get_dataset(name: str, *args, **kwargs):
    try:
        ds_info = get_dataset_info(name)
    except ValueError as e:
        raise ValueError(f"unknown dataset: {name}")
        
    return Dataset(*args, **kwargs,
                   normalize=ds_info["normalize"],
                   filename_template=get_filename_template(name)
                  )