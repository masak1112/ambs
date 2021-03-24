from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Yan Ji"
__date__ = "2020-11-10"

import argparse
import os
import numpy as np
import tensorflow as tf
import warnings
import pickle
from random import seed
import datetime
import json
from netCDF4 import Dataset,date2num
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from normalization import Norm_data
from metadata import MetaData as MetaData
from main_scripts.main_train_models import *
from data_preprocess.preprocess_data_step2 import *
import shutil
from model_modules.video_prediction import datasets, models, metrics


class Postprocess(TrainModel,ERA5Pkl2Tfrecords):
    def __init__(self, input_dir=None, results_dir=None, checkpoint=None, mode="test",
                      batch_size=None, num_samples=None, num_stochastic_samples=1, stochastic_plot_id=0,
                      gpu_mem_frac=None, seed=None,args=None):
        """
        The function for inference, generate results and images
        input_dir     :str, The root directory of tfrecords
        results_dir   :str, The output directory to save results
        checkpoint    :str, The directory point to the checkpoints
        mode          :str, Default is test, could be "train","val", and "test"
        batch_size    :int, The batch size used for generating test samples for each iteration
        num_samples   :int, The number of test samples used for generating output.
                            The maximum values should be the total number of samples for test dataset
        num_stochastic_samples: int, for the stochastic models such as SAVP, VAE, it is used for generate a number of
                                     ensemble for each prediction.
                                     For deterministic model such as convLSTM, it is default setup to 1
        stochastic_plot_id :int, the index for stochastically generated images to plot
        gpu_mem_frac       :int, GPU memory fraction to be used
        seed               :seed for control test samples
        """
     
        self.input_dir = os.path.normpath(input_dir)
        self.results_dir = self.output_dir = os.path.normpath(results_dir) 
        if not os.path.exists(self.results_dir):os.makedirs(self.results_dir)
        self.batch_size = batch_size
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.num_samples = num_samples
        self.num_stochastic_samples = num_stochastic_samples
        self.stochastic_plot_id = stochastic_plot_id
        self.input_dir_tfrecords = os.path.join(self.input_dir, "tfrecords")
        self.input_dir_pkl = os.path.join(self.input_dir, "pickle")
        self.args = args 
        self.checkpoint = checkpoint
        self.mode = mode
        if self.num_samples < self.batch_size: raise ValueError("The number of samples should be at least as large as the batch size. Currently, number of samples: {} batch size: {}".format(self.num_samples, self.batch_size))
        if checkpoint is None: raise ("The directory point to checkpoint is empty, must be provided for postprocess step")     
    

    def __call__(self):
        self.set_seed()
        self.get_metadata()
        self.copy_data_model_json()
        self.save_args_to_option_json()
        self.load_jsons()
        self.setup_test_dataset()
        self.setup_model()
        self.get_data_params()
        self.setup_num_samples_per_epoch()
        self.get_coordinates()
        self.get_stat_file()
        self.make_test_dataset_iterator() 
        self.check_stochastic_samples_ind_based_on_model()
        self.setup_graph()
        self.setup_gpu_config()
       
                
    def copy_data_model_json(self):
        """
        Copy the datasplit_conf.json model.json files from checkpoints directory to results_dir
        """
        if os.path.isfile(os.path.join(self.checkpoint,"options.json")):
            shutil.copy(os.path.join(self.checkpoint,"options.json"),
                        os.path.join(self.results_dir,"options_checkpoints.json"))
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,"options.json")))
        if os.path.isfile(os.path.join(self.checkpoint,"dataset_hparams.json")):
            shutil.copy(os.path.join(self.checkpoint,"dataset_hparams.json"),
                        os.path.join(self.results_dir,"dataset_hparams.json"))
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,"dataset_hparams.json"))) 

        if os.path.isfile(os.path.join(self.checkpoint,"model_hparams.json")):
            shutil.copy(os.path.join(self.checkpoint,"model_hparams.json"),
                        os.path.join(self.results_dir,"model_hparams.json"))
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,
                                                                                     "model_hparams.json")))

        if os.path.isfile(os.path.join(self.checkpoint,"data_dict.json")):
            shutil.copy(os.path.join(self.checkpoint,"data_dict.json"), os.path.join(self.results_dir,"data_dict.json"))
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,"data_dict.json")))


    def save_args_to_option_json(self):
        """
        Save the argments defined by user to the results dir
        """
    
        with open(os.path.join(self.results_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(self.args), sort_keys=True, indent=4))


    def load_jsons(self):
        """
        Copy all the jsons files that contains the data and model configurations 
        from the results directory for furthur usage
        """
        self.datasplit_dict = os.path.join(self.results_dir,"data_dict.json")
        self.model_hparams_dict = os.path.join(self.results_dir,"model_hparams.json")
        with open(os.path.join(self.results_dir, "options_checkpoints.json")) as f:
            self.options_checkpoint = json.loads(f.read())
            self.dataset = self.options_checkpoint["dataset"]
            self.model = self.options_checkpoint["model"]    
        self.model_hparams_dict_load = self.get_model_hparams_dict()    

    def setup_test_dataset(self):
        """
        setup the test dataset instance
        """
        VideoDataset = datasets.get_dataset_class(self.dataset)
        self.test_dataset = VideoDataset(input_dir=self.input_dir,mode=self.mode,datasplit_config=self.datasplit_dict)
        
    def setup_num_samples_per_epoch(self):
        """
        For generating images, the user can define the examples used, and will be taken as num_examples_per_epoch
        For testing we only use exactly one epoch, but to be consistent with the training, we keep the name '_per_epoch'
        """
        if self.num_samples:
            if self.num_samples > self.test_dataset.num_examples_per_epoch():
                raise ValueError('num_samples cannot be larger than the dataset')
            self.num_samples_per_epoch = self.num_samples
        else:
            self.num_samples_per_epoch = self.test_dataset.num_examples_per_epoch()
        return self.num_samples_per_epoch
   
    def get_data_params(self):
        """
        Get the context_frames, future_frames and total frames from hparamters settings.
        Note that future_frames_length is the number of predicted frames.
        """
        self.context_frames = self.model_hparams_dict_load["context_frames"]
        self.sequence_length = self.model_hparams_dict_load["sequence_length"]
        self.future_length = self.sequence_length - self.context_frames 

    def get_coordinates(self):
        """
        Retrieves the latitudes and longitudes from the metadata json file.
        """
        metadata_fname = os.path.join(self.input_dir,"metadata.json")
        md = MetaData(json_file=metadata_fname)
        md.get_metadata_from_file(metadata_fname)
        try:
            self.lats = md.lat
            self.lons = md.lon
            return md.lat, md.lon
        except:
            raise ValueError("Error when handling latitude and longitude in: '"+metadata_fname+"'")

    def get_stat_file(self):
        """
        Load the statistics from statistic file from the input directory
        """
        self.stat_fl = os.path.join(self.input_dir,"pickle/statistics.json")
 


    def make_test_dataset_iterator(self):
        """
        Make the dataset iterator
        """
        self.test_tf_dataset = self.test_dataset.make_dataset(self.batch_size)
        self.test_iterator = self.test_tf_dataset.make_one_shot_iterator()
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.test_handle = self.test_iterator.string_handle()
        self.iterator = tf.data.Iterator.from_string_handle(
            self.test_handle, self.test_tf_dataset.output_types, self.test_tf_dataset.output_shapes)
        self.inputs = self.iterator.get_next()
        self.input_ts = self.inputs["T_start"]
        #if self.dataset == "era5" and self.model == "savp":
        #   del self.inputs["T_start"]


    def check_stochastic_samples_ind_based_on_model(self):
        """
        stochastic forecasting only suitable for the geneerate models such as SAVP, vae. 
        For convLSTM, McNet only do determinstic forecasting
        """
        if self.model == "convLSTM" or self.model == "test_model" or self.model == 'mcnet':
            if self.num_stochastic_samples > 1:
                print("Number of samples for deterministic model cannot be larger than 1. Higher values are ignored.")
            self.num_stochastic_samples = 1
    
    def init_session(self):
        self.sess = tf.Session(config = self.config)
        self.sess.graph.as_default()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


    def run_and_plot_inputs_per_batch(self):
        """
        batch_id :int, the index in each batch size, maximum value is the batch_size
        """
        self.input_results = self.sess.run(self.inputs)
        self.input_images = self.input_results["images"]
        self.t_starts_results = self.input_results["T_start"]
        print("t_starts_results:",self.t_starts_results)
        self.t_starts = self.t_starts_results
        #get one seq and the corresponding start time poin
        #self.t_starts = self.input_results["T_start"]
        self.input_images_denorm_all = []
        for batch_id in range(self.batch_size):
            self.input_images_ = Postprocess.get_one_seq_from_batch(self.input_images,batch_id)
            #Renormalized data for inputs
            ts = Postprocess.generate_seq_timestamps(self.t_starts[batch_id],len_seq=self.sequence_length)
            input_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl,self.input_images_,self.vars_in)
            assert len(input_images_denorm.shape) == 4
            Postprocess.plot_seq_imgs(imgs = input_images_denorm[self.context_frames:,:,:,0],lats=self.lats,lons=self.lons,ts=ts[self.context_frames:],label="Ground Truth",output_png_dir=self.results_dir)
            self.input_images_denorm_all.append(list(input_images_denorm))
        assert len(np.array(self.input_images_denorm_all).shape) == 5
        return self.input_results, self.input_images_denorm_all,self.t_starts


    def run_stochastic(self):
        """
        Run session, save results to netcdf, plot input images, generate images and persistent images
        """
        self.init_session()     
        self.restore(self.sess, self.checkpoint)
        #Loop for samples
        self.sample_ind = 0
        self.persistent_loss_all_batches = []  # store the evaluation metric with shape [future_len]
        self.stochastic_loss_all_batches = []  # store the stochastic model metric with shape [stochastic,batch, future_len]
        while self.sample_ind < self.num_samples_per_epoch:
            if self.num_samples_per_epoch < self.sample_ind:
                break
            else:
                self.input_results, self.input_images_denorm_all, self.t_starts = self.run_and_plot_inputs_per_batch() #run the inputs and plot each sequence images
            
            feed_dict = {input_ph: self.input_results[name] for name, input_ph in self.inputs.items()}
            gen_loss_stochastic_batch = [] #[stochastic_ind,future_length]
            gen_images_stochastic = []   #[stochastic_ind,batch_size,seq_len,lat,lon,channels]
            #Loop for stochastics 
            for stochastic_sample_ind in range(self.num_stochastic_samples):
                print("stochastic_sample_ind:",stochastic_sample_ind)
                gen_images = self.sess.run(self.video_model.outputs['gen_images'], feed_dict=feed_dict)#return [batchsize,seq_len,lat,lon,channel]
                assert gen_images.shape[1] == self.sequence_length - 1 #The generate images seq_len should be sequence_len -1, since the last one is not used for comparing with groud truth
                gen_images_per_batch = []
                if stochastic_sample_ind == 0: 
                    persistent_images_per_batch = [] #[batch_size,seq_len,lat,lon,channel]
                    ts_batch = [] 
                for i in range(self.batch_size):
                    # generate time stamps for sequences only once, since they are the same for all ensemble members
                    if stochastic_sample_ind == 0: 
                        self.ts = Postprocess.generate_seq_timestamps(self.t_starts[i], len_seq=self.sequence_length)
                        init_date_str = self.ts[0].strftime("%Y%m%d%H")
                        ts_batch.append(init_date_str)
                        # get persistence_images
                        self.persistence_images, self.ts_persistence = Postprocess.get_persistence(self.ts,self.input_dir_pkl)
                        persistent_images_per_batch.append(self.persistence_images)
                        assert len(np.array(persistent_images_per_batch).shape) == 5 
                        self.plot_persistence_images()
                                
                    # Denormalized data for generate
                    gen_images_ = gen_images[i]
                    self.gen_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl, gen_images_, self.vars_in)
                    gen_images_per_batch.append(self.gen_images_denorm)
                    assert len(np.array(gen_images_per_batch).shape) == 5 
                    # only plot when the first stochastic ind otherwise too many plots would be created
                    # only plot the stochastic results of user-defined ind
                    self.plot_generate_images(stochastic_sample_ind, self.stochastic_plot_id)
                #calculate the persistnet error per batch
                if stochastic_sample_ind == 0:
                   persistent_loss_per_batch = Postprocess.calculate_metrics_by_batch(self.input_images_denorm_all,persistent_images_per_batch,self.future_length,self.context_frames,matric="mse",channel=0)
                   self.persistent_loss_all_batches.append(persistent_loss_per_batch)
                   
                #calculate the gen_images_per_batch error
                gen_loss_per_batch =  Postprocess.calculate_metrics_by_batch(self.input_images_denorm_all,gen_images_per_batch,self.future_length,self.context_frames,matric="mse",channel=0)                   
                gen_loss_stochastic_batch.append(gen_loss_per_batch) # self.gen_images_stochastic[stochastic,future_length]
                print("gen_images_per_batch shape:",np.array(gen_images_per_batch).shape)
                gen_images_stochastic.append(gen_images_per_batch)# [stochastic,batch_size, seq_len, lat, lon, channel]
                
                 #Switch the 0 and 1 psition
                print("before transpose:",np.array(gen_images_stochastic).shape)
            gen_images_stochastic = np.transpose(np.array(gen_images_stochastic),(1,0,2,3,4,5)) #[batch_size, stochastic, seq_len, lat, lon, chanel]
            Postprocess.check_gen_images_stochastic_shape(gen_images_stochastic)         
            assert len(gen_images_stochastic.shape) == 6
            assert np.array(gen_images_stochastic).shape[1] == self.num_stochastic_samples
            
            self.stochastic_loss_all_batches.append(gen_loss_stochastic_batch) #[samples/batch_size,stochastic,future_length]
            # save input and stochastic generate images to netcdf file
            # For each prediction (either deterministic or ensemble) we create one netCDF file.
            for batch_id in range(self.batch_size):
                self.save_to_netcdf_for_stochastic_generate_images(self.input_images_denorm_all[batch_id], persistent_images_per_batch[batch_id],
                                                            np.array(gen_images_stochastic)[batch_id], 
                                                            fl_name="vfp_date_{}_sample_ind_{}.nc".format(ts_batch[batch_id],self.sample_ind+batch_id))
            
            self.sample_ind += self.batch_size
        
        self.persistent_loss_all_batches = np.mean(np.array(self.persistent_loss_all_batches),axis=0)
        self.stochastic_loss_all_batches = np.mean(np.array(self.stochastic_loss_all_batches),axis=0)
        assert len(np.array(self.persistent_loss_all_batches).shape) == 1 
        assert np.array(self.persistent_loss_all_batches).shape[0] == self.future_length
        print("Bug here:",np.array(self.stochastic_loss_all_batches).shape)
        assert len(np.array(self.stochastic_loss_all_batches).shape) == 2
        assert np.array(self.stochastic_loss_all_batches).shape[0] == self.num_stochastic_samples
        
    
  
   
    def run_deterministic(self):
        """
        This function run the detereminstic forecasting and calculate the evaluation metric for deterministic model
        """
        self.init_session()
        self.restore(self.sess, self.checkpoint)
        #Loop for samples
        self.sample_ind = 0
        self.init_eval_metrics_list()
        while self.sample_ind < self.num_samples_per_epoch:
            if self.num_samples_per_epoch < self.sample_ind:
                break
            else:
                self.input_results, self.input_images_denorm_all, self.t_starts = self.run_and_plot_inputs_per_batch() #run the inputs and plot each sequence images
            feed_dict = {input_ph: self.input_results[name] for name, input_ph in self.inputs.items()}

            gen_images = self.sess.run(self.video_model.outputs['gen_images'], feed_dict=feed_dict)#return [batchsize,seq_len,lat,lon,channel]
            assert gen_images.shape[1] == self.sequence_length - 1 #The generate images seq_len should be sequence_len -1, since the last one is not used for comparing with groud truth 
            
            for i in range(self.batch_size):
                #get persistent prediction per sample
                self.get_and_plot_persistent_per_sample(sample_id=i)
                
                #get model prediction per sample 
                gen_images_ = gen_images[i]
                self.gen_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl, gen_images_, self.vars_in)
                self.plot_generate_images(0, 0)
                
                #save each sample of persistent, model forecasting and reference to netcdf file
                self.save_to_netcdf_for_stochastic_generate_images(self.input_images_denorm_all[i], self.persistence_images,
                                                            np.expand_dims(np.array(self.gen_images_denorm), axis=0),
                                                            fl_name="vfp_date_{}_sample_ind_{}.nc".format(self.ts_persistence[self.context_frames-1:self.context_frames][0].strftime("%Y%m%d%H"), self.sample_ind+i))

                #calculate the evaluation metric for persistent and model forecasting per sample
                self.calculate_persistence_eval_metrics(i)
                self.calculate_generate_eval_metrics(i)
 
            self.sample_ind += self.batch_size
        
        self.average_eval_metrics_for_all_batches()        
        self.turn_deter_to_stochastic() 


    def init_eval_metrics_list(self):
        """
        Initilizat all the metrics list to store the evaluation results
        """
        self.persistent_loss_all_batches = []  # store the evaluation metric with shape [future_len]
        self.gen_loss_all_batches = []  # store the determinstic model metric with shape [future_len]
        self.persistent_loss_all_batches_psnr = []
        self.gen_loss_all_batches_psnr = []


    def calculate_persistence_eval_metrics(self, i):
        # calculate the evaluation metric for persistent and model forecasting per sample
        persistent_loss_per_sample = Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i], self.persistence_images, self.future_length, self.context_frames, metric="mse", channel=0)
        self.persistent_loss_all_batches.append(persistent_loss_per_sample)
        persistent_loss_per_sample_psnr = Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i], self.persistence_images, self.future_length, self.context_frames, metric="psnr", channel=0)
        self.persistent_loss_all_batches_psnr.append(persistent_loss_per_sample_psnr)

    def calculate_generate_eval_metrics(self, i):
        """
        Calculate evaluation metrics for generate models
        """
        gen_loss_per_sample =Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i], self.gen_images_denorm, self.future_length, self.context_frames, metric="mse",channel=0)
        self.gen_loss_all_batches.append(gen_loss_per_sample)
        gen_loss_per_sample_psnr=Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i], self.gen_images_denorm, self.future_length, self.context_frames, metric="psnr",channel=0)
        self.gen_loss_all_batches_psnr.append(gen_loss_per_sample_psnr)

    def average_eval_metrics_for_all_batches(self):
        """
        average evaluation metrics for all the samples
        """
        self.persistent_loss_all_batches = np.mean(np.array(self.persistent_loss_all_batches), axis=0)
        self.persistent_loss_all_batches_psnr = np.mean(np.array(self.persistent_loss_all_batches_psnr), axis=0)
       
        self.gen_loss_all_batches = np.mean(np.array(self.gen_loss_all_batches), axis=0)
        self.gen_loss_all_batches_psnr = np.mean(np.array(self.gen_loss_all_batches_psnr), axis=0)


    def turn_deter_to_stochastic(self):
        self.stochastic_loss_all_batches =  np.expand_dims(self.gen_loss_all_batches, axis=0) #[1,future_lenght]
        self.stochastic_loss_all_batches_psnr = np.expand_dims(self.gen_loss_all_batches_psnr, axis=0) #[1,future_lenght]

    def get_and_plot_persistent_per_sample(self, sample_id):
        """
        Function that get persistent predictoin per sample and plot them 
        """
        self.ts = Postprocess.generate_seq_timestamps(self.t_starts[sample_id], len_seq=self.sequence_length)
        self.init_date_str = self.ts[0].strftime("%Y%m%d%H")
        # get persistence_images
        self.persistence_images, self.ts_persistence = Postprocess.get_persistence(self.ts, self.input_dir_pkl)
        print ("self.persistentc_images shape:", self.persistence_images.shape[0])
        assert self.persistence_images.shape[0] == self.sequence_length - 1
        self.plot_persistence_images()


    def run(self):
        if self.model == "convLSTM" or self.model == "test_model" or self.model == 'mcnet':
            self.run_deterministic()
        else:
            self.run_stochastic()
    
    
    def calculate_metrics(self,metric="mse",by=0):
        """ 
        args:
             mse: str the metric type, mse
             by: which channel of output based on to calculate error
        Calculate the mes metrics
        return a dictionary
        eval_metrics = {
                         "model_ts_{t1}":[stochast_error1,stochast_err2,.....]
                         "model_ts_{t2}": [stochast_erro1,stochast_err2,.....]
                         "persisent_ts_{t1}":[determinstic_error1]
                         "persistent_ts_{t2}":[determinstic_error1]
                       }
        {t1} is the forecasting timestamp
        """
        self.input_images_denorm_all_batches = np.array(self.input_images_denorm_all_batches)
        self.persistent_images_all_batches = np.array(self.persistent_images_all_batches)
        self.stochastic_images_all_batches = np.array(self.stochastic_images_all_batches)
        assert len(self.input_images_denorm_all_batches.shape) == 5
        assert len(self.stochastic_images_all_batches.shape) == 6
        assert len(self.persistent_images_all_batches.shape) == 5
        self.eval_metrics = {}
        for ts in range(self.future_length):
            #calcualte the metric on persistent
            mse_persistent =  (np.square(self.input_images_denorm_all_batches[:,self.context_frames+ts,:,:,by] -  self.persistent_images_all_batches[:,self.context_frames+ts-1,:,:,by])).mean()
            self.eval_metrics["persistent_ts_"+str(ts)] = [mse_persistent]
            self.stochastic_evals = []
            for stochastic_sample_ind in range(self.num_stochastic_samples):
                mse_model = (np.square(self.input_images_denorm_all_batches[:,self.context_frames+ts,:,:,by]- self.stochastic_images_all_batches[:,stochastic_sample_ind,self.context_frames+ts-1,:,:,by])).mean()
                self.stochastic_evals.append(str(mse_model))
                self.eval_metrics["model_ts_"+str(ts)] = self.stochastic_evals
        print("metric",self.eval_metrics)
        with open (os.path.join(self.results_dir,metric),"w") as fjs:
            json.dump(self.eval_metrics,fjs)
    
    @staticmethod
    def calculate_metrics_by_batch(input_per_batch,output_per_batch,future_length,context_frames, metric="mse", channel=0):
        """
        Calculate the metrics by samples per batch
        args:
	     input_per_batch : list or array, shape is [batch_size, seq_len,lat,lon,channel], seq_len is the sum of context_frames and future_length, the references input
             output_per_batch: list or array, shape is [batch_size,seq_len-1,lat,lon,channel],seq_len for output_per_batch is 1 less than the input_per_batch, the forecasting outputs
             future_lengths:   int, the future frames to be predicted
             context_frames:   int, the inputs frames used as input to the model
             matric:       :   str, the metric evaluation type
             channel       :   int, the channel of output which is used for calculating the metrics 
        return:
             loss : a list with length of future_length  
        """
        input_per_batch = np.array(input_per_batch)
        output_per_batch = np.array(output_per_batch)
        assert len(input_per_batch.shape) == 5
        assert len(output_per_batch.shape)  == 5
        eval_metrics_by_ts = []
        for ts in range(future_length):
            if metric == "mse":
                loss  =  (np.square(input_per_batch[:,context_frames+ts,:,:,channel] -  output_per_batch[:,context_frames+ts-1,:,:,channel])).mean()
            eval_metrics_by_ts.append(loss)
        assert len(eval_metrics_by_ts) == future_length
        return eval_metrics_by_ts
    
    @staticmethod
    def calculate_metrics_by_sample(input_per_sample, output_per_sample, future_length, context_frames, metric, channel):
        input_per_sample = np.array(input_per_sample)
        output_per_sample = np.array(output_per_sample)
        eval_metrics_by_ts = []
        for ts in range(future_length):
            if metric == "mse":
                loss = (np.square(input_per_sample[context_frames+ts, :, :, channel] -  output_per_sample[context_frames+ts-1, :, :, channel])).mean()
            elif metric == "psnr":
                loss = metrics.psnr_imgs(input_per_sample[context_frames+ts, :, :, channel], output_per_sample[context_frames+ts-1, :, :, channel])
            else:
                raise ValueError("We currently only support metric 'mse' and  'psnr' as evaluation metric for detereminstic forecasting")
            eval_metrics_by_ts.append(loss)
        return eval_metrics_by_ts

    def save_one_eval_metric_to_json(self,metric="mse"):
        """
        save list to pickle file in results directory
        """
        self.eval_metrics = {}
        if metric == "mse" :
            stochastic_loss_all_batches = self.stochastic_loss_all_batches #mse loss
        elif metric == "psnr" :
            stochastic_loss_all_batches = self.stochastic_loss_all_batches_psnr #psnr_loss
        else:
            raise ValueError(
                "We currently only support metric 'mse' and  'psnr' as evaluation metric for detereminstic forecasting")
        for ts in range(self.future_length):
            self.eval_metrics["persistent_ts_"+str(ts)] =  [self.persistent_loss_all_batches[ts]]
            #for stochastic_sample_ind in range(self.num_stochastic_samples): 
            self.eval_metrics["model_ts_"+str(ts)] = [str(i) for i in stochastic_loss_all_batches[:, ts]]
        with open (os.path.join(self.results_dir, metric), "w") as fjs:
            json.dump(self.eval_metrics, fjs)

    def save_eval_metric_to_json(self):
        """
        Save all the evaluation metrics to the json file
        """
        self.save_one_eval_metric_to_json(metric="mse")
        self.save_one_eval_metric_to_json(metric="psnr")


    @staticmethod
    def check_gen_images_stochastic_shape(gen_images_stochastic):
        """
        For models with deterministic forecasts, one dimension would be lacking. Therefore, here the array
        dimension is expanded by one.
        """
        if len(np.array(gen_images_stochastic).shape) == 6:
            pass
        elif len(np.array(gen_images_stochastic).shape) == 5:
            gen_images_stochastic = np.expand_dims(gen_images_stochastic, axis=0)
        else:
            raise ValueError("Passed gen_images_stochastic  is not of the right shape")
        return gen_images_stochastic

    @staticmethod
    def denorm_images(stat_fl,input_images_,channel,var):
        """
        denormaize one channel of images for particular var
        args:
            stat_fl       : str, the path of the statistical json file
            input_images_ : list/array [seq, lat,lon,channel], the input images are  denormalized
            channel       : the channel of images going to be denormalized
            var           : the variable name of the channel, 

        """
        norm_cls  = Norm_data(var)
        norm = 'minmax' # TODO: can be replaced by loading option.json from previous step, if this information is saved there.
        with open(stat_fl) as js_file:
             norm_cls.check_and_set_norm(json.load(js_file),norm)
        input_images_denorm = norm_cls.denorm_var(input_images_[:, :, :,channel], var, norm)
        return input_images_denorm

    @staticmethod
    def denorm_images_all_channels(stat_fl,input_images_,vars_in):
        """
        Denormalized all the channles of images
        args:
            stat_fl       : str, the path of the statistical json file
            input_images_ : list/array [seq, lat,lon,channel], the input images are  denormalized
            vars_in       : list of str, the variable names of all the channels
        """
        
        input_images_all_channles_denorm = []
        input_images_ = np.array(input_images_)
        
        for c in range(len(vars_in)):
            input_images_all_channles_denorm.append(Postprocess.denorm_images(stat_fl,input_images_,channel=c,var=vars_in[c]))           
        input_images_denorm = np.stack(input_images_all_channles_denorm, axis=-1)
        return input_images_denorm
    
    @staticmethod
    def get_one_seq_from_batch(input_images,i):
        """
        Get one sequence images from batch images
        """
        assert (len(np.array(input_images).shape)==5)
        input_images_ = input_images[i, :, :, :, :]
        return input_images_

    @staticmethod
    def generate_seq_timestamps(t_start, len_seq=20):

        """
        Given the start timestampe and generate the len_seq hourly sequence timestamps
        
        args:
            t_start   :int, str, array, the defined start timestamps
            len_seq   :int, the sequence length for generating hourly timestamps
        """
        if isinstance(t_start,int): t_start = str(t_start)
        if isinstance(t_start,np.ndarray):
            warnings.warn("You give array of timestamps, we only use the first timestamp as start datetime to generate sequence timestamps")
            t_start = str(t_start[0])
        if not len(t_start) == 10:
            raise ValueError ("The timestamp gived should following the pattern '%Y%m%d%H' : 2017121209")
        s_datetime = datetime.datetime.strptime(t_start, '%Y%m%d%H')
        seq_ts = [s_datetime + datetime.timedelta(hours = i+1) for i in range(len_seq)]
        return seq_ts

    def plot_persistence_images(self):
        """
        Plot the persistence images
        """
       # I am not sure about the number of frames given with context_frames and context_frames +
        Postprocess.plot_seq_imgs(imgs=self.persistence_images[self.context_frames-1:,:,:,0], lats=self.lats, lons=self.lons,
                                  ts=self.ts_persistence[self.context_frames-1:], label="Persistence Forecast" + self.model,output_png_dir=self.results_dir) 

    def plot_generate_images(self,stochastic_sample_ind,stochastic_plot_id=0):
        """
        Plot the generate image for specific stochastic index
        """
        if stochastic_sample_ind == stochastic_plot_id: 
            Postprocess.plot_seq_imgs(imgs=self.gen_images_denorm[self.context_frames-1:,:,:,0], lats=self.lats, lons=self.lons,
                                      ts=self.ts_persistence[self.context_frames-1:], label="Forecast by Model " + self.model, output_png_dir=self.results_dir)
        else:
            pass

    def save_to_netcdf_for_stochastic_generate_images(self, input_images_, persistent_images_, gen_images_stochastic, fl_name="test.nc"):
        """
        Save the input images, persistent images and generated stochatsic images to netCDF file
        args:
            input_images_        : list/array, [seq,lat,lon,channel]
            persistent_images_   : list/array, [seq-1,lat,lon,channel]
            gen_images_stochastic: list/array (float), [stochastic_number,seq-1,lat,lon,channel]
            fl_name              : str, the netcdf file name to be saved
        """
        print("inputs fpor netcdf:",input_images_)
        input_images_ = np.array(input_images_)
        persistent_images_ = np.array(persistent_images_)
        gen_images_stochastic = np.array(gen_images_stochastic)

        assert (len(np.array(input_images_).shape)==len(np.array(gen_images_stochastic).shape))-1
        assert len(persistent_images_.shape) == 4 #[seq,lat,lon,channel]
        y_len = len(self.lats)
        x_len = len(self.lons)
        ts_input = self.ts[:self.context_frames]
        ts_forecast = self.ts[self.context_frames:]
        gen_images_ = np.array(gen_images_stochastic)
        output_file = os.path.join(self.results_dir, fl_name)
        with Dataset(output_file, "w", format="NETCDF4") as nc_file:
            nc_file.title = 'ERA5 hourly reanalysis data and the forecasting data by deep learning for 2-m above sea level temperatures'
            nc_file.author = "Bing Gong, Michael Langguth"
            nc_file.create_date = "2020-08-04"
            #create groups forecasts and analysis 
            fcst = nc_file.createGroup("forecasts")
            analgrp = nc_file.createGroup("analysis")
            #create dims for all the data(forecast and analysis)
            latD = nc_file.createDimension('lat', y_len)
            lonD = nc_file.createDimension('lon', x_len)
            timeD = nc_file.createDimension('time_input', self.context_frames) 
            timeF = nc_file.createDimension('time_forecast', self.future_length)
            #Latitude
            lat  = nc_file.createVariable('lat', float, ('lat',), zlib = True)
            lat.units = 'degrees_north'
            lat[:] = self.lats
            #Longitude
            lon = nc_file.createVariable('lon', float, ('lon',), zlib = True)
            lon.units = 'degrees_east'
            lon[:] = self.lons

            #Time for input
            time = nc_file.createVariable('time_input', 'f8', ('time_input',), zlib = True)
            time.units = "hours since 1970-01-01 00:00:00" 
            time.calendar = "gregorian"
            time[:] = date2num(ts_input, units = time.units, calendar = time.calendar)
        
            #time for forecast
            time_f = nc_file.createVariable('time_forecast', 'f8', ('time_forecast',), zlib = True)
            time_f.units = "hours since 1970-01-01 00:00:00" 
            time_f.calendar = "gregorian"
            time_f[:] = date2num(ts_forecast, units = time.units, calendar = time.calendar)
        
           ################ analysis group  #####################
        
            #####sub group for inputs
            # create variables for non-meta data
            #Temperature
            t2 = nc_file.createVariable("/analysis/inputs/T2","f4",("time_input","lat","lon"), zlib = True)
            t2.units = 'K'
            t2[:,:,:] = input_images_[:self.context_frames,:,:,0]

            #mean sea level pressure
            msl = nc_file.createVariable("/analysis/inputs/MSL","f4",("time_input","lat","lon"), zlib = True)
            msl.units = 'Pa'
            msl[:,:,:] = input_images_[:self.context_frames,:,:,1]

            #Geopotential at 500 
            gph500 = nc_file.createVariable("/analysis/inputs/GPH500","f4",("time_input","lat","lon"), zlib = True)
            gph500.units = 'm'
            gph500[:,:,:] = input_images_[:self.context_frames,:,:,2]
        
            #####sub group for reference(ground truth)
            #Temperature
            t2_r = nc_file.createVariable("/analysis/reference/T2","f4",("time_forecast","lat","lon"), zlib = True)
            t2_r.units = 'K'
            t2_r[:,:,:] = input_images_[self.context_frames:,:,:,0]

             #mean sea level pressure
            msl_r = nc_file.createVariable("/analysis/reference/MSL","f4",("time_forecast","lat","lon"), zlib = True)
            msl_r.units = 'Pa'
            msl_r[:,:,:] = input_images_[self.context_frames:,:,:,1]

            #Geopotential at 500 
            gph500_r = nc_file.createVariable("/analysis/reference/GPH500","f4",("time_forecast","lat","lon"), zlib = True)
            gph500_r.units = 'm'
            gph500_r[:,:,:] = input_images_[self.context_frames:,:,:,2]

            ###subgroup for Pesistent analysis #######
            t2_p = nc_file.createVariable("/analysis/persistent/T2","f4",("time_forecast","lat","lon"), zlib = True)
            t2_p.units = 'K'
            t2_p[:,:,:] = persistent_images_[self.context_frames-1:,:,:,0]

            #msl_p = nc_file.createVariable("/analysis/persistent/MSL","f4",("time_forecast","lat","lon"), zlib = True)
            #msl_p.units = 'Pa'
            #msl_p[:,:,:] = persistent_images_[self.context_frames:,:,:,1]
             
            #Geopotential at 500 
            #gph500_p = nc_file.createVariable("/analysis/persistent/GPH500","f4",("time_forecast","lat","lon"), zlib = True)
            #gph500_p.units = 'm'
            #gph500_p[:,:,:] = persistent_images_[self.context_frames:,:,:,2]


            ################ forecast group  #####################
            for stochastic_sample_ind in range(self.num_stochastic_samples):
                #Temperature:
                t2 = nc_file.createVariable("/forecasts/T2/stochastic/{}".format(stochastic_sample_ind),"f4",("time_forecast","lat","lon"), zlib = True)
                t2.units = 'K'
                t2[:,:,:] = gen_images_[stochastic_sample_ind,self.context_frames-1:,:,:,0]

                #mean sea level pressure
                msl = nc_file.createVariable("/forecasts/MSL/stochastic/{}".format(stochastic_sample_ind),"f4",("time_forecast","lat","lon"), zlib = True)
                msl.units = 'Pa'
                msl[:,:,:] = gen_images_[stochastic_sample_ind,self.context_frames-1:,:,:,1]

                #Geopotential at 500 
                gph500 = nc_file.createVariable("/forecasts/GPH500/stochastic/{}".format(stochastic_sample_ind),"f4",("time_forecast","lat","lon"), zlib = True)
                gph500.units = 'm'
                gph500[:,:,:] = gen_images_[stochastic_sample_ind,self.context_frames-1:,:,:,2]        

            print("{} created".format(output_file)) 
        return None
    
    @staticmethod
    def plot_seq_imgs(imgs,lats,lons,ts,output_png_dir,label="Ground Truth"):
        """
        Plot the seq images 
        """
        if len(np.array(imgs).shape)!=3:raise("img dims should be four: (seq_len,lat,lon)")
        if np.array(imgs).shape[0]!= len(ts): raise("The len of timestamps should be equal the image seq_len") 
        fig = plt.figure(figsize=(18,6))
        gs = gridspec.GridSpec(1, len(ts))
        gs.update(wspace = 0., hspace = 0.)
        xlables = [round(i,2) for i  in list(np.linspace(np.min(lons),np.max(lons),5))]
        ylabels = [round(i,2) for i  in list(np.linspace(np.max(lats),np.min(lats),5))]
        for i in range(len(ts)):
            t = ts[i]
            ax1 = plt.subplot(gs[i])
            plt.imshow(imgs[i] ,cmap = 'jet', vmin=270, vmax=300)
            #plt.imshow(imgs[i] ,cmap = 'jet')
            ax1.title.set_text("t = " + t.strftime("%Y%m%d%H"))
            plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])
            if i == 0:
                plt.setp([ax1], xticks = list(np.linspace(0, len(lons), 5)), xticklabels = xlables, yticks = list(np.linspace(0, len(lats), 5)), yticklabels = ylabels)
                plt.ylabel(label, fontsize=10)
        plt.savefig(os.path.join(output_png_dir, label + "_TS_" + str(ts[0]) + ".jpg"))
        plt.clf()
        plt.close()
        output_fname = label + "_TS_" + ts[0].strftime("%Y%m%d%H") + ".jpg"
        print("image {} saved".format(output_fname))

    
    @staticmethod
    def get_persistence(ts,input_dir_pkl):
        """This function gets the persistence forecast.
        'Today's weather will be like yesterday's weather.'
    
        Inputs:
        ts: output by generate_seq_timestamps(t_start,len_seq=sequence_length)
            Is a list containing dateime objects
        
        input_dir_pkl: input directory to pickle files
    
        Ouputs:
        time_persistence:    list containing the dates and times of the 
                       persistence forecast.
        var_peristence  : sequence of images corresponding to the times
                       in ts_persistence
        """
        ts_persistence = []
        year_origin = ts[0].year
        for t in range(len(ts)): # Scarlet: this certainly can be made nicer with list comprehension 
            ts_temp = ts[t] - datetime.timedelta(days=1)
            ts_persistence.append(ts_temp)
        t_persistence_start = ts_persistence[0]
        t_persistence_end = ts_persistence[-1]
        year_start = t_persistence_start.year #Bing to address the issue #43 and Scarelet please confirm this change
        month_start = t_persistence_start.month
        month_end = t_persistence_end.month
        print("start year:",year_start)    
        # only one pickle file is needed (all hours during the same month)
        if month_start == month_end: 
            # Open files to search for the indizes of the corresponding time
            time_pickle  = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'T'))
            # Open file to search for the correspoding meteorological fields
            var_pickle  = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'X'))
            
            if year_origin != year_start:
               time_origin_pickle = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_origin, 12, 'T'))
               var_origin_pickle  = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_origin, 12, 'X'))            
               time_pickle.extend(time_origin_pickle)
               var_pickle.extend(var_origin_pickle)
            
           # Retrieve starting index
            ind = list(time_pickle).index(np.array(ts_persistence[0]))
            #print('Scarlet, Original', ts_persistence)
            #print('From Pickle', time_pickle[ind:ind+len(ts_persistence)])
        
            var_persistence  = np.array(var_pickle)[ind:ind+len(ts_persistence)]
            time_persistence = np.array(time_pickle)[ind:ind+len(ts_persistence)].ravel()
            #print(' Scarlet Shape of time persistence',time_persistence.shape)
            #print(' Scarlet Shape of var persistence',var_persistence.shape)
    
    
        # case that we need to derive the data from two pickle files (changing month during the forecast periode)
        else: 
            t_persistence_first_m  = [] # should hold dates of the first month
            t_persistence_second_m = [] # should hold dates of the second month
        
            for t in range(len(ts)):
                m = ts_persistence[t].month
                if m == month_start:
                    t_persistence_first_m.append(ts_persistence[t])
                if m == month_end:
                    t_persistence_second_m.append(ts_persistence[t])
            if year_origin == year_start: 
                # Open files to search for the indizes of the corresponding time
                time_pickle_first  = Postprocess.load_pickle_for_persistence(input_dir_pkl,year_start, month_start, 'T')
                time_pickle_second = Postprocess.load_pickle_for_persistence(input_dir_pkl,year_start, month_end, 'T')
        
                # Open file to search for the correspoding meteorological fields
                var_pickle_first  =  Postprocess.load_pickle_for_persistence(input_dir_pkl,year_start, month_start, 'X')
                var_pickle_second =  Postprocess.load_pickle_for_persistence(input_dir_pkl,year_start, month_end, 'X')
           
            if year_origin != year_start:
                # Open files to search for the indizes of the corresponding time
                time_pickle_second = Postprocess.load_pickle_for_persistence(input_dir_pkl,year_origin, 1, 'T')
                time_pickle_first = Postprocess.load_pickle_for_persistence(input_dir_pkl,year_start, 12, 'T')

                # Open file to search for the correspoding meteorological fields
                var_pickle_second  =  Postprocess.load_pickle_for_persistence(input_dir_pkl,year_origin, 1, 'X')
                var_pickle_first =  Postprocess.load_pickle_for_persistence(input_dir_pkl,year_start, 12, 'X')
                
                #print('Scarlet, Original', ts_persistence)
                #print('From Pickle', time_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)], time_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)])
                #print(' Scarlet before', time_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)].shape, time_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)].shape)
            
            # Retrieve starting index
            ind_first_m = list(time_pickle_first).index(np.array(t_persistence_first_m[0]))
            print ("time_pickle_second:",time_pickle_second)
            ind_second_m = list(time_pickle_second).index(np.array(t_persistence_second_m[0]))
        
             # append the sequence of the second month to the first month
            var_persistence  = np.concatenate((var_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)], 
                                          var_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)]), 
                                          axis=0)
            time_persistence = np.concatenate((time_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)],
                                          time_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)]), 
                                          axis=0).ravel() # ravel is needed to eliminate the unnecessary dimension (20,1) becomes (20,)
            #print(' Scarlet concatenate and ravel (time)', var_persistence.shape, time_persistence.shape)
           
        if len(time_persistence.tolist()) == 0 : raise ("The time_persistent is empty!")    
        if len(var_persistence) ==0 : raise ("The var persistence is empty!")
        # tolist() is needed for plottingi
        var_persistence = var_persistence[1:]
        time_persistence = time_persistence[1:]
        return var_persistence, time_persistence.tolist()
    
    @staticmethod
    def load_pickle_for_persistence(input_dir_pkl,year_start, month_start, pkl_type):
        """Helper to get the content of the pickle files. There are two types in our workflow:
        T_[month].pkl where the time stamp is stored
        X_[month].pkl where the variables are stored, e.g. temperature, geopotential and pressure
        This helper function constructs the directory, opens the file to read it, returns the variable. 
        """
        path_to_pickle = input_dir_pkl+'/'+str(year_start)+'/'+pkl_type+'_{:02}.pkl'.format(month_start)
        infile = open(path_to_pickle,'rb')    
        var = pickle.load(infile)
        return var

    def plot_evalution_metrics(self):
        model_names = self.eval_metrics.keys()
        model_ts_errors = [] # [timestamps,stochastic_number]
        persistent_ts_errors = []
        for ts in range(self.future_length-1):
            stochastic_err = self.eval_metrics["model_ts_"+str(ts)]  
            stochastic_err = [float(item) for item in stochastic_err]
            model_ts_errors.append(stochastic_err)  
            persistent_err = self.eval_metrics["persistent_ts_"+str(ts)]
            persistent_err = float(persistent_err[0])
            persistent_ts_errors.append(persistent_err)
        if len(np.array(model_ts_errors).shape) == 1:  model_ts_errors = np.expand_dims(np.array(model_ts_errors), axis=1)
        model_ts_errors = np.array(model_ts_errors)
        persistent_ts_errors = np.array(persistent_ts_errors)
        fig = plt.figure(figsize=(6,4))
        ax = plt.axes([0.1, 0.15, 0.75, 0.75])
        for stoch_ind in range(len(model_ts_errors[0])):
            plt.plot(model_ts_errors[:,stoch_ind],lw=1)
        plt.plot(persistent_ts_errors)
        plt.xticks(np.arange(1,self.future_length))
        ax.set_ylim(0., 10)
        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Time stamps')
        ax.set_ylabel("Errors")
        print("Saving plot for err")
        plt.savefig(os.path.join(self.results_dir,"evaluation.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type = str, required = True,
                        help = "either a directory containing subdirectories "
                               "train, val, test, etc, or a directory containing "
                               "the tfrecords")
    parser.add_argument("--results_dir", type = str, default = 'results',
                        help = "ignored if output_gif_dir is specified")
    parser.add_argument("--checkpoint",
                        help = "directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--mode", type = str, choices = ['train','val', 'test'], default = 'test',
                        help = 'mode for dataset, val or test.')
    parser.add_argument("--batch_size", type = int, default = 8, help = "number of samples in batch")
    parser.add_argument("--num_samples", type = int, help = "number of samples in total (all of them by default)")
    parser.add_argument("--num_stochastic_samples", type = int, default = 1)
    parser.add_argument("--stochastic_plot_id", type = int, default = 0, help = "The stochastic generate images index to plot")
    parser.add_argument("--gpu_mem_frac", type = float, default = 0.95, help = "fraction of gpu memory to use")
    parser.add_argument("--seed", type = int, default = 7)
    args = parser.parse_args()

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    test_instance = Postprocess(input_dir=args.input_dir,results_dir=args.results_dir,checkpoint=args.checkpoint,mode="test",
                      batch_size=args.batch_size,num_samples=args.num_samples,num_stochastic_samples=args.num_stochastic_samples,
                      gpu_mem_frac=args.gpu_mem_frac,seed=args.seed,stochastic_plot_id=args.stochastic_plot_id,args=args)

    test_instance()
    test_instance.run()
    test_instance.save_eval_metric_to_json()
    test_instance.plot_evalution_metrics() 
if __name__ == '__main__':
    main()
