
__email__ = "b.gong@fz-juelich.de"

from video_prediction.datasets.era5_dataset import *
import pytest
import xarray as xr

input_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/test_data_roshni"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/test/cv_test.json"
hparams_dict_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams_template.json"
mode = "val"


@pytest.fixture(scope="module")

def era5_dataset_case1():
    return ERA5Dataset(input_dir=input_dir, datasplit_config=datasplit_config, hparams_dict_config=hparams_dict_config,
                 mode="train", seed=1234, nsamples_ref=1000)

def test_init_era5_dataset(era5_dataset_case1):

    assert era5_dataset_case1.max_epochs == 20
    assert era5_dataset_case1.mode == mode
    assert era5_dataset_case1.batch_size == 4


def test_get_filenames_from_datasplit(era5_dataset_case1):
    era5_dataset_case1.get_filename_from_datasplit()
    flname="era5_vars4ambs_201801.nc"
    check = flname in era5_dataset_case1.filenames
    assert check == True

# def test_load_data_from_nc(era5_dataset_case1):
#     era5_dataset_case1.load_data_from_nc()
#     ds = xr.open_mfdataset(era5_dataset_case1.filenames)



