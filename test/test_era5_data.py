from video_prediction.datasets.era5_dataset import *
import pytest
import numpy as np
import json
import datetime

input_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2010toY2222M01to12-160x128-2970N1500W-T2_MSL_gph500/"
output_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test/tfrecords/"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
hparams_path = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams.json"

def get_model_hparams_dict(model_hparams_dict_path):
    """
    Get model_hparams_dict from json file
    """
    model_hparams_dict_load = {}
    if model_hparams_dict_path:
        with open(model_hparams_dict_path) as f:
            model_hparams_dict_load.update(json.loads(f.read()))
    return model_hparams_dict_load

model_hparams_dict = get_model_hparams_dict(hparams_path)

#generate an instance
@pytest.fixture(scope="module")
def era5_dataset_case1(input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,hparams_dict=model_hparams_dict,sequences_per_file=128,vars_in=["T2","MSL","gph500"]):
    
    return ERA5Pkl2Tfrecords(input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,hparams_dict=hparams_dict,sequences_per_file=sequences_per_file,vars_in=vars_in)


def test_get_metadata(era5_dataset_case1):
    """
    Test if the meta data extracted properly
    """
    assert era5_dataset_case1.height == 160
    assert era5_dataset_case1.width == 128
    assert era5_dataset_case1.vars_in == ["T2","MSL","gph500"]

def test_get_datasplit(era5_dataset_case1):
    #assert type of hparams
    assert input_dir == era5_dataset_case1.input_dir
    d = era5_dataset_case1.get_datasplit()

def test_get_months(era5_dataset_case1):
    assert len(era5_dataset_case1.get_months()) == 7

def test_parse_hparams(era5_dataset_case1):
    """
    Test the updated hparam is properly updated 
    """
    print("hparmas:",era5_dataset_case1.hparams)
    assert era5_dataset_case1.hparams.max_epochs == 20
    assert era5_dataset_case1.sequence_length == 20
    assert era5_dataset_case1.hparams.batch_size == 4

def test_save_tf_record(era5_dataset_case1):
    #create a sequence
    def create_n_dimensional_matrix(n):
            return [[[[[0 for i in range(n[4])] for j in range(n[3])] for k in range(n[2])] for l in range(n[1])] for h in range(n[0])]
    sequences = create_n_dimensional_matrix([10,20,64,64,3])
    sequences = np.array(sequences)
    #sequences[0,0,1,1,0] = 3
    #sequences[0,0,1,1,4] = 2
    out_fname = "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test/tfrecords/test1.tfrecord"
    #generate 10 timestamps
    t_start_points = [datetime.datetime.strptime(str(i),"%Y%m%d%H") for i in range(2020010100,2020010110)]
    print("t_start_points len:",len(t_start_points))
    era5_dataset_case1.save_tf_record(out_fname, sequences, t_start_points)


# # def test_read_pkl_and_save_tfrecords(era5_dataset_case1):
# #     print("var in:",era5_dataset_case1.vars_in)
# #     era5_dataset_case1.read_pkl_and_save_tfrecords(year=2017,month=9)
# #     #assert the input folder is proper
# #     assert era5_dataset_case1.input_file_year=="/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2010toY2222M01to12-160x128-2970N1500W-T2_MSL_gph500/pickle/2017"
    
    
@pytest.fixture(scope="module")
def era5_dataset_case2(seed=1234,input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,hparams_dict=model_hparams_dict,sequences_per_file=128,vars_in=["T2","MSL","gph500"]):
    return ERA5Dataset(seed=seed,input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,hparams_dict=hparams_dict,sequences_per_file=sequences_per_file,vars_in=vars_in)  


def test_init_era5_dataset(era5_dataset_case2):
    assert era5_dataset_case2.max_epochs == 20

def test_get_tfrecords_filesnames(era5_dataset_case2):
    era5_dataset_case2.get_tfrecords_filesnames()
    assert era5_dataset_case2.filenames[0] ==  "sequence_Y_2017_M_9_0_to_127.tfrecords"

# def test_check_pkl_tfrecords_consistency(era5_dataset_case1):
#     pass