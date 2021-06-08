
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Yanji"
__date__ = "2020-11-22"

from main_scripts.main_visualize_postprocess import *
import pytest
import numpy as np
import datetime
from netCDF4 import Dataset, date2num

########Test case 1################
results_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/results/era5-Y2007-2019M01to12-92x56-3840N0000E-2t_tcc_t_850/savp/20210324T120926_ji4_savp_cv12" 
checkpoint = "/p/project/deepacf/deeprain/video_prediction_shared_folder/models/era5-Y2007-2019M01to12-92x56-3840N0000E-2t_tcc_t_850/savp/20210324T120926_ji4_savp_cv12" 
mode = "test"
batch_size = 2
num_stochastic_samples = 2
gpu_mem_frac = 0.5
seed = 12345
eval_metrics=["mse", "psnr"]


class MyClass:
    def __init__(self, i):
         self.input_dir = i
         self.dataset = "era5"
         self.model = "test_model"
args = MyClass(results_dir)

#####instance1###
@pytest.fixture(scope="module")
def vis_case1():
    return Postprocess(results_dir=results_dir,checkpoint=checkpoint,
                       mode=mode,batch_size=batch_size, 
                       num_stochastic_samples=num_stochastic_samples,
                       seed=seed,args=args,eval_metrics=eval_metrics)

def test_load_jsons(vis_case1):
    assert vis_case1.dataset == "era5"
    assert vis_case1.model == "savp"
    assert vis_case1.input_dir_tfr == "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2007-2019M01to12-92x56-3840N0000E-2t_tcc_t_850/tfrecords_seq_len_24"
    assert vis_case1.run_mode == "deterministic"

def test_get_metadata(vis_case1):
    assert vis_case1.height == 56
    assert vis_case1.width == 92
    assert vis_case1.vars_in[0] == "2t"
    assert vis_case1.vars_in[1] == "tcc"


def test_setup_test_dataset(vis_case1):
    vis_case1.test_dataset.mode == mode

def test_get_data_params(vis_case1):
    assert vis_case1.context_frames == 12
    assert vis_case1.future_length == 12

def test_run_deterministic(vis_case1):
    vis_case1.num_samples_per_epoch = 20
    vis_case1.init_session()
    vis_case1.restore(vis_case1.sess,vis_case1.checkpoint)
    print("fcast-product",vis_case1.fcst_products)
    eval_metric_ds = Postprocess.init_metric_ds(vis_case1.fcst_products, vis_case1.eval_metrics, vis_case1.vars_in[vis_case1.channel], vis_case1.num_samples_per_epoch, vis_case1.future_length)

    input_results,input_images_denorm_all,t_starts = vis_case1.get_input_data_per_batch(vis_case1.inputs) 
    assert len(t_starts) == batch_size
    ts_1 = t_starts[0][0]
    year = str(ts_1)[:4]
    month = str(ts_1)[4:6]
    filename = "ecmwf_era5_" +  str(ts_1)[2:] + ".nc"
    fl = os.path.join("/p/scratch/deepacf/deeprain/ambs_era5/extractedData",year, month, filename)
    print("netCDF file name:",fl)
    with Dataset(fl,"r")  as data_file:
       t2_var = data_file.variables["2t"][0,:,:]
    t2_var = np.array(t2_var)    
    t2_max = np.max(t2_var[117:173,0:92])
    t2_min = np.min(t2_var[117:173,0:92])
    input_image = np.array(input_images_denorm_all)[0,0,:,:,0] #get the first batch id and 1st sequence image
    input_img_max = np.max(input_image)
    input_img_min = np.min(input_image)
    print("input_image",input_image[0,:10])
    assert t2_max == input_img_max
    assert t2_min == input_img_min
    sample_ind = 0 
    feed_dict = {input_ph: input_results[name] for name, input_ph in vis_case1.inputs.items()}
    gen_images = vis_case1.sess.run(vis_case1.video_model.outputs['gen_images'], feed_dict=feed_dict)
    gen_images_denorm = vis_case1.denorm_images_all_channels(gen_images, vis_case1.vars_in, vis_case1.norm_cls,
                                                                norm_method="minmax")
    ############Test persistenct value#############
    times_0, init_times = vis_case1.get_init_time(t_starts)
    batch_ds = vis_case1.create_dataset(input_images_denorm_all, gen_images_denorm, init_times)
    nbs = np.minimum(vis_case1.batch_size, vis_case1.num_samples_per_epoch - sample_ind)
  
    times_seq = (pd.date_range(times_0[0], periods=int(vis_case1.sequence_length), freq="h")).to_pydatetime() 
    persistence_seq, _ = Postprocess.get_persistence(times_seq, vis_case1.input_dir_pkl)
    ts_1_per = (pd.to_datetime(times_0[0]) -  datetime.timedelta(hours=23)).strftime("%Y%m%d%H")
    
    year_per = str(ts_1_per)[:4]
    month_per = str(ts_1_per)[4:6]
    filename_per = "ecmwf_era5_" +  str(ts_1_per)[2:] + ".nc"
 
    fl_per = os.path.join("/p/scratch/deepacf/deeprain/ambs_era5/extractedData",year_per,month_per,filename_per)
    with Dataset(fl_per,"r")  as data_file:
        t2_var_per = data_file.variables["2t"][0,117:173,0:92]    
     
    t2_per_var = np.array(t2_var_per)
    t2_per_max = np.max(t2_per_var)
    per_image_max = np.max(persistence_seq[0])
    assert t2_per_max == per_image_max
    

    ##Test evaluation metric
    for ivar, var in enumerate(vis_case1.vars_in):
        batch_ds["{0}_persistence_fcst".format(var)].loc[dict(init_time=init_times[0])] = \
                        persistence_seq[vis_case1.context_frames-1:, :, :, ivar]
        
    eval_metric_ds = vis_case1.populate_eval_metric_ds(eval_metric_ds,batch_ds,sample_ind,vis_case1.vars_in[vis_case1.channel])
    ##now manuly calculate the mse and see if values is the same as the ones in eval_metric_ds
    #calculate the mse between generateed images and reference images
    sample_gen = gen_images_denorm[0,vis_case1.context_frames-1:,:,:,vis_case1.channel]  
    sample_ref = input_images_denorm_all[0,vis_case1.context_frames:,:,:,vis_case1.channel]
    sample_gen_ref_mse_t0 = np.mean((sample_gen[0] - sample_ref[0])**2)
    metric_name = "2t_savp_mse"
    print("eval_metric_ds",eval_metric_ds)
    assert eval_metric_ds[metric_name][0,0] == sample_gen_ref_mse_t0
    sample_gen_ref_mse_t5 = np.mean((sample_gen[5] - sample_ref[5])**2)
    assert eval_metric_ds[metric_name][0,5] == sample_gen_ref_mse_t5   



       


    
       

#def test_run_determinstic_quantile_plot(vis_case1):
#    vis_case1.init_metric_ds()



#def test_make_test_dataset_iterator(vis_case1):
#    vis_case1.make_test_dataset_iterator()
#    pass


#def test_check_stochastic_samples_ind_based_on_model(vis_case1):
#    vis_case1.check_stochastic_samples_ind_based_on_model()
#    assert vis_case1.num_stochastic_samples == 1


#def test_run_and_plot_inputs_per_batch(vis_case1):
#    """
#    Test we get the right datasplit data
#    """
#    vis_case1.get_stat_file()
#    vis_case1.setup_gpu_config()
#    vis_case1.init_session()
#    vis_case1.run_and_plot_inputs_per_batch()
#    test_datetime = vis_case1.t_starts[0][0]
#    test_datetime = datetime.datetime.strptime(str(test_datetime), "%Y%m%d%H")
#    assert test_datetime.month == 3



#def test_run_test(vis_case1):
#    """
#    Test the running on test dataset
#    """
#    vis_case1()
#    vis_case1.run()






