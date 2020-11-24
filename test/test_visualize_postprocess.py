

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

@pytest.fixture(scope="module")
def vis_case1():
    return Postprocess(input_dir=input_dir,results_dir=results_dir,checkpoint=checkpoint,
                       mode=mode,batch_size=batch_size,num_samples=num_samples,num_stochastic_samples=num_stochastic_samples,
                       gpu_mem_frac=gpu_mem_frac,seed=seed,args=args)

def test_get_metadata(vis_case1):
    vis_case1.get_metadata()
    assert vis_case1.vars_in[0] == "T2"
    assert vis_case1.vars_in[1] == "MSL"

def copy_
def test_load_params_from_checkpoints_dir(vis_case1):
    vis_case1.load_params_from_checkpoints_dir()
    
