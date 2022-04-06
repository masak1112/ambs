
__email__ = "b.gong@fz-juelich.de"

from video_prediction.datasets.gzprcp_dataset import *
import pytest
import xarray as xr

input_dir = "/p/largedata/jjsc42/project/deeprain/project_data/10min_AWS_prcp"
datasplit_config = "/p/project/deepacf/deeprain/ji4/ambs/video_prediction_tools/data_split/gzprcp/datasplit.json"
hparams_dict_config = "/p/project/deepacf/deeprain/ji4/ambs/video_prediction_tools/hparams/gzprcp_data/convLSTM_gan/model_hparams_template.json"
sequences_per_file = 10
mode = "val"


@pytest.fixture(scope="module")
def gzprcp_dataset_case1():
    dataset =  GzprcpDataset(input_dir=input_dir, datasplit_config=datasplit_config, hparams_dict_config=hparams_dict_config,
                 mode="val", seed=1234, nsamples_ref=1000)
    print('***********ok*************')
    dataset.get_hparams()
    return dataset

def test_init_gzprcp_dataset(gzprcp_dataset_case1):
    # gzprcp_dataset_case1.get_hparams()
    print('gzprcp_dataset_case1.max_epochs: {}'.format(gzprcp_dataset_case1.max_epochs))
    print('gzprcp_dataset_case1.mode: {}'.format(gzprcp_dataset_case1.mode))
    print('gzprcp_dataset_case1.batch_size: {}'.format(gzprcp_dataset_case1.batch_size))
    print('gzprcp_dataset_case1.k: {}'.format(gzprcp_dataset_case1.k))
    assert gzprcp_dataset_case1.max_epochs == 8
    assert gzprcp_dataset_case1.mode == mode
    assert gzprcp_dataset_case1.batch_size == 32
    assert gzprcp_dataset_case1.k == 0.001

'''
def test_get_filenames_from_datasplit(era5_dataset_case1):
    era5_dataset_case1.get_filename_from_datasplit()
    flname="era5_vars4ambs_201801.nc"
    check = flname in era5_dataset_case1.filenames
    assert check  == True




def test_load_data_from_nc(era5_dataset_case1):
    era5_dataset_case1.load_data_from_nc()
    ds = xr.open_mfdataset(era5_dataset_case1.filenames)

'''

