
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-11-22"

from main_scripts.main_visualize_postprocess import *
import pytest
import numpy as np
import datetime

input_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5_test" 
results_dir = "/p/home/jusers/gong1/juwels/video_prediction_shared_folder/results/era5_test"
checkpoint = "/p/home/jusers/gong1/juwels/video_prediction_shared_folder/models/test/test_model/20201110T1301_gong1"
mode = "test"
batch_size = 16
num_samples = 16
num_stochastic_samples = 2
gpu_mem_frac = 0.5
seed=12345

class MyClass:
    def __init__(self, i):
         self.input_dir = i
         self.dataset = "era5"
         self.model = "test_model"
args = MyClass(input_dir)

#####instance1###
@pytest.fixture(scope="module")
def vis_case1():
    return Postprocess(input_dir=input_dir,results_dir=results_dir,checkpoint=checkpoint,
                       mode=mode,batch_size=batch_size,num_samples=num_samples,num_stochastic_samples=num_stochastic_samples,
                       gpu_mem_frac=gpu_mem_frac,seed=seed,args=args)
######instance2
num_samples2 = 200000
@pytest.fixture(scope="module")
def vis_case2():
    return Postprocess(input_dir=input_dir,results_dir=results_dir,checkpoint=checkpoint,
                       mode=mode,batch_size=batch_size,num_samples=num_samples2,num_stochastic_samples=num_stochastic_samples,
                       gpu_mem_frac=gpu_mem_frac,seed=seed,args=args)

def test_get_metadata(vis_case1):
    vis_case1.get_metadata()
    assert vis_case1.vars_in[0] == "T2"
    assert vis_case1.vars_in[1] == "MSL"

def test_copy_data_model_json(vis_case1):
    vis_case1.copy_data_model_json()
    isfile_copy = os.path.isfile(os.path.join(checkpoint,"options.json"))
    assert isfile_copy == True
    isfile_copy_model_hpamas = os.path.isfile(os.path.join(checkpoint,"model_hparams.json"))
    assert isfile_copy_model_hpamas == True

def test_load_json(vis_case1):
    vis_case1.load_jsons()
    assert vis_case1.dataset == "era5"
    assert vis_case1.model == "test_model"

def test_setup_num_samples_per_epoch(vis_case1):
    vis_case1.load_jsons()
    vis_case1.setup_test_dataset()
    vis_case1.setup_num_samples_per_epoch()  
    assert vis_case1.num_samples_per_epoch == 16
    
def test_get_data_params(vis_case1):
    vis_case1.get_data_params()
    assert vis_case1.context_frames == 10
    assert vis_case1.future_frames == 10


def test_get_coordinates(vis_case1):
    vis_case1.get_coordinates()
    assert len(vis_case1.lats) == 128


def test_make_test_dataset_iterator(vis_case1):
    vis_case1.make_test_dataset_iterator()
    pass


def test_check_stochastic_samples_ind_based_on_model(vis_case1):
    vis_case1.check_stochastic_samples_ind_based_on_model()
    assert vis_case1.num_stochastic_samples == 1


def test_run_and_plot_inputs_per_batch(vis_case1):
    """
    Test we get the right datasplit data
    """
    vis_case1.get_stat_file()
    vis_case1.setup_gpu_config()
    vis_case1.init_session()
    vis_case1.run_and_plot_inputs_per_batch()
    test_datetime = vis_case1.t_starts[0][0]
    test_datetime = datetime.datetime.strptime(str(test_datetime), "%Y%m%d%H")
    assert test_datetime.month == 3



def test_run_test(vis_case1):
    """
    Test the running on test dataset
    """
    vis_case1()
    vis_case1.run()

