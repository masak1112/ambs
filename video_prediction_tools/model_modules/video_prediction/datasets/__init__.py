#from .base_dataset import BaseVideoDataset
#from .era5_dataset import ERA5Dataset
#from .gzprcp_dataset import GzprcpDataset
#from .moving_mnist import MovingMnist
#from data_preprocess.dataset_options import known_datasets
from .stats import MinMax, ZScore
from .dataset import Dataset
import dask
from dask.base import tokenize
from data_preprocess.dataset_options import known_datasets as advertised_datasets

# store any relevant Dataset data
known_datasets = {
    "era5": MinMax,
    "weatherbench": ZScore
}

DATE_TEMPLATE = "{year}-{month:02d}"

def get_filename_template(name):
    return f"{name}_{DATE_TEMPLATE}.nc"


def get_dataset(name: str, *args, **kwargs):
    if name not in known_datasets.keys():
        raise ValueError(f"unknown dataset: {name}")
    
    return Dataset(*args, **kwargs,
                   normalize=known_datasets[name],
                   filename_template=get_filename_template(name)
                  )

if advertised_datasets != set(known_datasets.keys()):
    raise Exception("dataset_options.known_datasets differ from datasets.known_datasets")