from video_prediction.datasets.era5_dataset import *
import pytest
import numpy as np
import json

path_to_datasplit = "../hparams/era5/${model}/model_hparams.json"

input_dir = "/p/home/jusers/gong1/juwels/video_prediction_shared_folder/preprocessedData/era5-Y2017to2017M01to12_wb025-160x128-2970N1500W-T2_MSL_gph500/pickle/2017_test"
output_dir = "/p/home/jusers/gong1/juwels/video_prediction_shared_folder/preprocessedData/era5-Y2017to2017M01to12_wb025-160x128-2970N1500W-T2_MSL_gph500/tfrecords/2017_test"
datasplit_config = "/p/home/jusers/gong1/juwels/ambs/video_prediction_tools/data_split/cv_test.json"

#generate an instance
@pytest.fixture(scope="module")
def era5_dataset_case1(input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,sequences_per_file=128):
    return ERA5Pkl2Tfrecords(input_dir,output_dir,datasplit_config,sequences_per_file)


def test_get_datasplit(era5_dataset_case1):
    #assert type of hparams
    assert input_dir == era5_dataset_case1.input_dir
    d = era5_dataset_case1.get_datasplit()
    #assert the element in the parsed hparams is as expected, to check is the hparams is updated as expected

def test_get_months(era5_dataset_case1):
    #print("months:",era5_dataset_case1.get_months())
    assert len(era5_dataset_case1.get_months()) == 7
    






