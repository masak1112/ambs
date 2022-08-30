#from .base_dataset import BaseVideoDataset
#from .era5_dataset import ERA5Dataset
#from .gzprcp_dataset import GzprcpDataset
#from .moving_mnist import MovingMnist
#from data_preprocess.dataset_options import known_datasets
from .stats import MinMax, ZScore
from .dataset import Dataset

known_datasets = {
    "era5": MinMax,
    "weatherbench": ZScore
}

filename_date_template = "_{year}-{month:02d}.nc"


def get_dataset(name: str, *args, **kwargs):
    if name not in known_datasets.keys():
        raise ValueError(f"unknown dataset: {name}")
    
    return Dataset(*args, **kwargs, normalize=known_datasets[name], filename_template=f"{name}{filename_date_template}")
    