

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"



from video_prediction.datasets.era5_dataset import *
import pytest
import numpy as np
import json
import datetime

input_dir =  "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
hparams_dict_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams.json"
sequences_per_file = 10
mode = "val"


@pytest.fixture(scope="module")
def era5_dataset_case2():
    return ERA5Dataset(input_dir=input_dir,mode=mode,
                       datasplit_config=datasplit_config,hparams_dict_config=hparams_dict_config,seed=1234)  

def test_init_era5_dataset(era5_dataset_case2):
    assert era5_dataset_case2.hparams.max_epochs == 20
    assert era5_dataset_case2.mode == mode

def test_get_tfrecords_filesnames(era5_dataset_case2):
    era5_dataset_case2.get_tfrecords_filesnames_base_datasplit()
    assert era5_dataset_case2.filenames[0] ==  os.path.join(input_dir,"tfrecords","sequence_Y_2017_M_2_0_to_9.tfrecords")# def test_check_pkl_tfrecords_consistency(era5_dataset_case1):

def test_get_example_info(era5_dataset_case2):
    era5_dataset_case2.get_tfrecords_filesnames_base_datasplit()
    era5_dataset_case2.get_example_info()
    assert era5_dataset_case2.image_shape[0] == 160
    assert era5_dataset_case2.image_shape[1] == 128
    assert era5_dataset_case2.image_shape[2] == 3



   
