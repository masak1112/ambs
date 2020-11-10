

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"



from video_prediction.datasets.era5_dataset import *
import pytest
import numpy as np
import json
import datetime

input_dir =  "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2010toY2222M01to12-160x128-2970N1500W-T2_MSL_gph500"
output_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test/tfrecords/"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
hparams_dict_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams.json"
vars_in = ["T2","MSL","gph500"]
sequences_per_file = 10


#generate an instance for ERA5Pkl2Tfrecords
@pytest.fixture(scope="module")
def era5_dataset_case1():
    return ERA5Pkl2Tfrecords(input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,
                             hparams_dict_config=hparams_dict_config,sequences_per_file=sequences_per_file)

    
def test_get_datasplit(era5_dataset_case1):
    assert input_dir == era5_dataset_case1.input_dir
    d = era5_dataset_case1.get_datasplit()

def test_get_months(era5_dataset_case1):
    assert len(era5_dataset_case1.get_years_months()[1]) == 3

def test_get_metadata(era5_dataset_case1):
    """
    Test if the meta data extracted properly
    """
    assert era5_dataset_case1.height == 160
    assert era5_dataset_case1.width == 128
    assert era5_dataset_case1.vars_in == ["T2","MSL","gph500"]

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


def test_read_pkl_and_save_tfrecords(era5_dataset_case1):
    print("var in:",era5_dataset_case1.vars_in)
    month_test = list(era5_dataset_case1.get_months())[1]
    #Bing comments the following since the test for even one month will take very long time
    #era5_dataset_case1.read_pkl_and_save_tfrecords(year=2017,month=month_test)
    #assert the input folder is proper
    #assert era5_dataset_case1.input_file_year=="/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2010toY2222M01to12-160x128-2970N1500W-T2_MSL_gph500/pickle/2017"
    #assert the output tfrecords is saved properly
    fname_case1 = "sequence_Y_2017_M_{}_0_to_{}.tfrecords".format(month_test,sequences_per_file-1)
    if_file_exit = os.path.isfile(os.path.join(output_dir,fname_case1))
    assert if_file_exit == True

#Generate instance for class ERA5Dataset
mode2 = "val"
@pytest.fixture(scope="module")
def era5_dataset_case2():
    return ERA5Dataset(seed=1234,input_dir=input_dir,mode=mode2,output_dir=output_dir,
                       datasplit_config=datasplit_config,hparams_dict_config=hparams_dict_config,
                       sequences_per_file=sequences_per_file)  

def test_init_era5_dataset(era5_dataset_case2):
    assert era5_dataset_case2.max_epochs == 20
    assert era5_dataset_case2.mode == mode

def test_get_tfrecords_filesnames(era5_dataset_case2):
    era5_dataset_case2.input_dir_tfrecords = output_dir
    era5_dataset_case2.get_tfrecords_filesnames_base_datasplit()
    assert era5_dataset_case2.filenames[0] ==  os.path.join(output_dir,"sequence_Y_2017_M_2_0_to_9.tfrecords")# def test_check_pkl_tfrecords_consistency(era5_dataset_case1):

def test_get_example_info(era5_dataset_case2):
    era5_dataset_case2.get_tfrecords_filesnames_base_datasplit()
    era5_dataset_case2.get_example_info()
    assert era5_dataset_case2.image_shape[0] == 160
    assert era5_dataset_case2.image_shape[1] == 128
    assert era5_dataset_case2.image_shape[2] == 3



   
