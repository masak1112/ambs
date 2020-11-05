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
def train_model_case1(input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,
                       model_hparams_dict=hparams_path,model=model,checkpoint=checkpoint,dataset=dataset,
                       gpu_mem_frac=gpu_mem_grac,seed=seed,args=args):
    return TrainModel(input_dir,output_dir,datasplit_config,
                       model_hparams_dict,model,checkpoint,dataset,
                        gpu_mem_frac,seed,args)

 #################################################### End Test Configuration for instance case 1################################################

def test_generate_output_dir(train_model_case1):
    assert train_model_case1.output_dir ==  output_dir


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
    assert train_model_case1.video_model.learning_rate == 0.001
    assert train_model_case1.video_model.loss_fun == "rmse"


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
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        val_handle_eval = sess.run(train_model_case1.val_handle)
        fetch_val = {}
        fetch_val["x"]  = train_model_case1.inputs["T_start"]
        val_results = sess.run(fetch_val,feed_dict={train_model_case1.train_handle: val_handle_eval})  
        val_t = val_results["x"]
        val_t1 = datetime.datetime.strptime(str(val_t[0][0]), "%Y%m%d%H")
        val_month = val_t1.month
        assert val_month  == 2

def test_save_dataset_model_params_to_checkpoint_dir(train_model_case1):
    """
    Test if all the args, model hparamters, data hparamters are saved properly to outoput directory
    """
    train_model_case1.save_dataset_model_params_to_checkpoint_dir()
    #check if options.json was stored in the right place
    if_option = os.path.isfile(os.path.join(train_model_case1.output_dir,"options.json"))
    assert if_option == True


def test_calculate_samples_and_epochs(train_model_case1):
    train_model_case1.calculate_samples_and_epochs()
    assert train_model_case1.num_examples == 680


def test_create_fetches_for_train(train_model_case1):
    if_train_op = False
    if_g_losses = False
    train_model_case1.setup_model()
    train_model_case1.make_dataset_iterator()
    train_model_case1.setup_graph()
    train_model_case1.create_fetches_for_train()
    assert len(list(train_model_case1.fetches.keys()))  == 4
    if "train_op" in list(train_model_case1.fetches.keys()):if_train_op = True 
    assert if_train_op == True
    if "g_losses" in list(train_model_case1.fetches.keys()):if_g_losses = True
    assert if_g_losses == False

def test_setup_graph(train_model_case1):
    #train_model_case1.setup_graph()
    assert train_model_case1.video_model.x == train_model_case1.inputs["images"]    
    assert train_model_case1.video_model.is_build_graph == True

def test_count_paramters(train_model_case1):
    #Give a simple example contains two trainable parameters for testing
    train_model_case1.count_parameters()
    train_model_case1.setup_gpu_config()
    global_step = tf.train.get_or_create_global_step()
    with tf.Session(config=train_model_case1.config) as sess:
        params = sess.run(train_model_case1.parameter_count)
    assert params == 2

def test_train_models(train_model_case1):
    train_model_case1.setup()
    train_model_case1.total_steps = 20
    train_model_case1.train_model()
    #check if the model is saved properly
    file_saved = os.path.join(output_dir,"model-19.meta")
    if_file_saved = os.path.isfile(file_saved)
    assert if_file_saved == True
