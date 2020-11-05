__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"
__date__ = "2020-10-22"


from main_scripts.main_train_models import *
import pytest
import numpy as np
import datetime

####################################################Start Test Configuration for instsance case 1################################################
input_dir =  "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test"
output_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/models/test"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
hparams_path = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams.json"
model = "test_model"
checkpoint = ""
dataset = "era5"
gpu_mem_grac = 0.9
seed = 1234
class MyClass:
    def __init__(self, i):
         self.input_dir = i
args = MyClass(input_dir)

@pytest.fixture(scope="module")
def train_model_case2(input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,
                       model_hparams_dict=hparams_path,model=model,checkpoint=checkpoint,dataset=dataset,
                       gpu_mem_frac=gpu_mem_grac,seed=seed,args=args):
    return TrainModel(input_dir,output_dir,datasplit_config,
                       model_hparams_dict,model,checkpoint,dataset,
                        gpu_mem_frac,seed,args)
