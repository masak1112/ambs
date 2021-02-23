__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2021-02-22"

from main_scripts.main_train_models import *
from model_modules.video_prediction.models.vanilla_vae_model import  *
import pytest
import numpy as np
import datetime

####################################################Start Test Configuration for instsance case 1################################################
input_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2009to2020M01to12-64x64-3930N0300E-T2_MSL_gph500/"
output_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/models/test"
datasplit_dict = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
model_hparams_dict = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/vae/model_hparams.json"
model = "vae"
checkpoint = ""
dataset = "era5"
gpu_mem_frac = 0.9
seed = 1234
class MyClass:
    def __init__(self, i):
         self.input_dir = i
         self.dataset = dataset
         self.model = model
args = MyClass(input_dir)


@pytest.fixture(scope="module")
def train_vae_model_case1(scope="module"):
    return TrainModel(input_dir,output_dir,datasplit_dict,
                       model_hparams_dict,model,checkpoint,dataset,
                       gpu_mem_frac,seed,args)


def test_get_model_hparams_dict(train_vae_model_case1):
    train_vae_model_case1.get_model_hparams_dict()
    assert train_vae_model_case1.dataset == "era5"


def test_train_model(train_vae_model_case1):
    train_vae_model_case1.set_seed()
    train_vae_model_case1.get_model_hparams_dict()
    train_vae_model_case1.load_params_from_checkpoints_dir()
    train_vae_model_case1.setup_dataset()
    train_vae_model_case1.setup_model()
    train_vae_model_case1.make_dataset_iterator()
    train_vae_model_case1.x = train_vae_model_case1.inputs["images"]
    train_vae_model_case1.global_step = tf.train.get_or_create_global_step()
    original_global_variables = tf.global_variables()
    q, z_mu, z_log_sigma_sq, z = VanillaVAEVideoPredictionModel.vae_arc3(train_vae_model_case1.x[:, 0, :, :, :], l_name=0, nz=16)    
    
    assert train_vae_model_case1.video_model.learning_rate == 0.001
    assert train_vae_model_case1.video_model.nz == 16
    assert train_vae_model_case1.video_model.loss_fun  == "rmse"
    assert train_vae_model_case1.video_model.shuffle_on_val == False
