
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"



from main_scripts.main_train_models import *
import pytest
import numpy as np

input_dir =  "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test"
output_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/models/test"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
hparams_path = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams.json"
model = "convLSTM"
checkpoint = ""
dataset = "era5"
gpu_mem_grac = 0.9
seed = 1234

@pytest.fixture(scope="module")
def train_model_case1(input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,
                       model_hparams_dict=hparams_path,model=model,checkpoint=checkpoint,dataset=dataset,
                       gpu_mem_frac=gpu_mem_grac,seed=seed):
    return TrainModel(input_dir,output_dir,datasplit_config,
                       model_hparams_dict,model,checkpoint,dataset,
                       gpu_mem_frac,seed)


def test_get_model_hparams_dict(train_model_case1):
    train_model_case1.get_model_hparams_dict()
    assert list(train_model_case1.model_hparams_dict_load.keys())[0] == "batch_size"
    assert train_model_case1.dataset == "era5"

def test_setup_dataset(train_model_case1):
    train_model_case1.setup_dataset()
    train_fnames = train_model_case1.train_dataset.filenames
    val_fnames = train_model_case1.val_dataset.filenames
    assert len(train_fnames) != 0
    assert len(val_fnames) != 0
    assert train_fnames[0]!=val_fnames[0]

def test_setup_model(train_model_case1):
    print("setup model:",train_model_case1.model_hparams_dict)
    train_model_case1.setup_model()
    assert train_model_case1.hparams_dict["context_frames"] == 10



def test_make_dataset_iterator(train_model_case1):
    train_model_case1.make_dataset_iterator()
    assert train_model_case1.batch_size == 4
    with tf.Session() as sess:
        sess.run
