
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"



from main_scripts.main_train_models import *
import pytest
import numpy as np
import datetime

####################################################Start Test Configuration################################################
input_dir =  "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test"
output_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/models/test"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
hparams_path = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams.json"
model = "convLSTM"
checkpoint = ""
dataset = "era5"
gpu_mem_grac = 0.9
seed = 1234
##################################################### End Test Configuration################################################


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
    """
    Check if the hparameters are updated properly
    """
    print("setup model:",train_model_case1.model_hparams_dict)
    train_model_case1.setup_model()
    assert train_model_case1.hparams_dict["context_frames"] == 10
    assert train_model_case1.model.learning_rate == 0.001
    assert train_model_case1.model.loss_fun == "rmse"



def test_make_dataset_iterator(train_model_case1):
    """
     Here to test, the training and valition dataset into the model/session should be the correct ones.
    """
    train_model_case1.make_dataset_iterator()
    assert train_model_case1.batch_size == 4
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        fetch = {}
        fetch["x"] = train_model_case1.inputs["T_start"]
        train_t_start = sess.run(fetch)
        print("train_t_start,",train_t_start)
        train_t = train_t_start["x"]
        train_t1 = datetime.datetime.strptime(str(train_t[0][0]), "%Y%m%d%H")
        train_month = train_t1.month 
        assert train_month  == 1
        #val_handle_eval = sess.run(self.val_handle) 


def test_make_dataset_iterator_for_val(train_model_case1):
    """
    This is to test if the validation iterator is setup properly that should be consistent with the dataplist config 
    """
    train_model_case1.make_dataset_iterator()
    with tf.Session() as sess:
        val_handle_eval = sess.run(train_model_case1.val_handle)
        
    
