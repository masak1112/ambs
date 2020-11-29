from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-11-10"

import argparse
import errno
import os
from os import path
import sys
import math
import random
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import pickle
from random import seed
import datetime
import json
from os.path import dirname
from netCDF4 import Dataset,date2num
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from video_prediction import datasets, models
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim
from normalization import Norm_data
from metadata import MetaData as MetaData
from main_scripts.main_train_models import *
from data_preprocess.preprocess_data_step2 import *
import shutil
from video_prediction import datasets, models

class Postprocess(TrainModel,ERA5Pkl2Tfrecords):
    def __init__(self,input_dir=None,results_dir=None,checkpoint=None,mode="test",
                      batch_size=None,num_samples=None,num_stochastic_samples=1,
                      gpu_mem_frac=None,seed=None,args=None):
        """
        The function for inference, generate results and images
        input_dir     :str, the root directory of tfrecords 
        results_dir   :str, the output directory to save results
        checkpoint    :str, the directory point to the checkpoints
        mode          :str, default is test, could be "train","val", and "test"
        dataset       :str, the dataset type, "era5","moving_mnist", or "kth"
        """
        #super(Postprocess,self).__init__(input_dir=input_dir,output_dir=None,datasplit_dir=data_split_dir,
        #                                  model_hparams_dict=model_hparams_dict,model=model,checkpoint=checkpoint,dataset=dataset,
        #                                  gpu_mem_frac=gpu_mem_frac,seed=seed,args=args)        
        self.input_dir = input_dir
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.num_samples = num_samples
        self.num_stochastic_samples = num_stochastic_samples
        self.input_dir_tfrecords = os.path.join(self.input_dir,"tfrecords")
        self.input_dir_pkl = os.path.join(self.input_dir,"pickle") 
        if checkpoint is None: raise ("The directory point to checkpoint is empty, must be provided for postprocess step")     
        self.args = args 
        self.checkpoint = checkpoint
        self.mode = mode

    def __call__(self):
        self.set_seed()
        self.get_metadata()#get the vars_in variables which contains the list of input variable names
        self.copy_data_model_json()
        self.load_json()
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
        self.initia_save_data()
       
                
    def copy_data_model_json(self):
        """
        Copy the datasplit_conf.json model.json files from checkpoints directory to results_dir
        """
        if os.path.isfile(os.path.join(self.checkpoint,"options.json")):
            shutil.copy(os.path.join(self.checkpoint,"options.json"), os.path.join(self.results_dir,"options_checkpoints.json"))                  
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,"options.json")))
        if os.path.isfile(os.path.join(self.checkpoint,"dataset_hparams.json")):
            shutil.copy(os.path.join(self.checkpoint,"dataset_hparams.json"), os.path.join(self.results_dir,"dataset_hparams.json"))
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,"dataset_hparams.json"))) 

        if os.path.isfile(os.path.join(self.checkpoint,"model_hparams.json")):
            shutil.copy(os.path.join(self.checkpoint,"model_hparams.json"), os.path.join(self.results_dir,"model_hparams.json"))
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,"model_hparams.json")))

        if os.path.isfile(os.path.join(self.checkpoint,"data_dict.json")):
            shutil.copy(os.path.join(self.checkpoint,"data_dict.json"), os.path.join(self.results_dir,"data_dict.json"))
        else:
            raise FileNotFoundError("the file {} does not exist".format(os.path.join(self.checkpoint,"data_dict.json")))

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
        """
        if self.num_samples:
            if self.num_samples > self.test_dataset.num_examples_per_epoch():
                raise ValueError('num_samples cannot be larger than the dataset')
            self.num_examples_per_epoch = self.num_samples
        else:
            self.num_examples_per_epoch = self.test_dataset.num_examples_per_epoch()
        return self.num_examples_per_epoch
   
    def get_data_params(self):
        """
        Get the context_frames, future_frames and total frames from hparamters settings.
        """
        self.context_frames = self.model_hparams_dict_load["context_frames"]
        self.sequence_length = self.model_hparams_dict_load["sequence_length"]
        self.future_frames = self.sequence_length - self.context_frames 

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
            raise ValueError("Error when handling: '"+metadata_fname+"'")

    def get_stat_file(self):
        """
        Load the statistic files from input directory
        """
        self.stat_fl = os.path.join(self.input_dir,"pickle/statistics.json")
 
    def initia_save_data(self):
        self.sample_ind = 0
        self.gen_images_all = []
        self.persistent_images_all = []
        self.input_images_all = []
        return self.sample_ind, self.gen_images_all, self.persistent_images_all, self.input_images_all


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
        if self.dataset == "era5" and self.model == "savp":
           del  self.inputs["T_start"]      


    def check_stochastic_samples_ind_based_on_model(self):
        if self.model == "convLSTM" or self.model == "test_model": self.num_stochastic_samples = 1
    
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
        #get one seq and the corresponding start time poin
        self.t_starts = self.input_results["T_start"]
        for batch_id in range(self.batch_size):
            self.input_images_ = Postprocess.get_one_seq_and_time_from_batch(self.input_images,self.t_starts,batch_id)
            #Renormalized data for inputs
            self.input_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl,self.input_images_,self.vars_in)
            Postprocess.plot_seq_imgs(imgs = self.input_images_denorm[self.context_frames+1:,:,:,0],lats=self.lats,lons=self.lons,ts=self.ts[self.context_frames+1:],label="Ground Truth",output_png_dir=self.results_dir)  
        return self.input_images

    def plot_persistence_images(self):
       # I am not sure about the number of frames given with context_frames and context_frames +
        Postprocess.plot_seq_imgs(imgs=self.persistence_images[self.context_frames+1:,:,:,0],lats=self.lats,lons=self.lons,ts=self.ts_persistence[self.context_frames+1:], label="Persistence Forecast" + self.model,output_png_dir=self.results_dir) 



    def plot_generate_images(self,stochastic_sample_ind):
           #Generate images inputs
        if stochastic_sample_ind == 0: 
          #Generate forecast images
            Postprocess.plot_seq_imgs(imgs=self.gen_images_denorm[self.context_frames:,:,:,0],lats=self.lats,lons=self.lons,ts=self.ts[self.context_frames+1:],label="Forecast by Model " + self.model,output_png_dir=self.results_dir) 
            #Generate persistent images
        else:
            pass

    def run(self):
        self.init_session()     
        self.restore(self.sess, args.checkpoint)
        #Loop for samples
        while self.sample_ind < self.num_samples_per_epoch:
            gen_images_stochastic = []
            if self.num_samples_per_epoch < self.sample_ind:
                break
            else:
                self.input_images = self.run_and_plot_inputs_per_batch() #run the inputs and plot each sequence images

            feed_dict = {input_ph: input_results[name] for name, input_ph in self.inputs.items()}
            gen_images_stochastic = [] #[stochastic_ind,batch_size,seq_len,lat,lon,channels]
            #Loop for stochastics 
            for stochastic_sample_ind in range(self.num_stochastic_samples):
                gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)#return [batchsize,seq_len,lat,lon,channel]
                assert gen_images.shape[1] == self.sequence_length-1 #The generate images seq_len should be sequence_len -1, since the last one is not used for comparing with groud truth 
                gen_images_per_batch = []
                ts_batch = [] 
                for i in range(self.batch_size):
                    #generate time stamps for sequences
                    if stochastic_sample_ind == 0: 
                        ts = Postprocess.generate_seq_timestamps(self.t_starts[i],len_seq=self.sequence_length)
                        init_date_str = ts[0].strftime("%Y%m%d%H")
                        ts_batch.append(init_date_str)
                        #only plot when the first stochastic ind 
                        self.plot_generate_images(stochastic_sample_ind)
                        #get persistence_images
                        self.persistence_images, self.ts_persistence = Postprocess.get_persistence(ts,self.input_dir_pkl)
                        self.plot_persistence_images()

                    #Renormalized data for generate
                    gen_images_ = gen_images[i]
                    gen_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl,gen_images_,self.vars_in)
                    gen_images_per_batch.append(gen_images_denorm)

            self.gen_images_stochastic.append(gen_images_per_batch)
            #save input and stochastic generate images to netcdf file
            for batch_id in range(self.batch_size):
                self.save_to_netcdf_for_stochastic_generate_images(input_images[batch_id],
                                                            gen_images_stochastic[:,batch_id,:,:,:,:],                                                                fl_name="vfp_date_{}_sample_ind_{}.nc".format(ts_batch[batch_id],sample_ind+batch_id))
            
            sample_ind += self.batch_size

    @staticmethod
    def denorm_images(stat_fl,input_images_,channel,var):
        norm_cls  = Norm_data(var)
        norm = 'minmax' #can be replaced by loading option.json from previous step
        with open(stat_fl) as js_file:
             norm_cls.check_and_set_norm(json.load(js_file),norm)
        input_images_denorm = norm_cls.denorm_var(input_images_[:, :, :,channel], var, norm)
        return input_images_denorm

    @staticmethod
    def denorm_images_all_channels(stat_fl,input_images_,*args):
        input_images_all_channles_denorm = []
        input_images_ = np.array(input_images_)
        args = [item for item in args][0]
        for c in range(len(args)):
            input_images_all_channles_denorm.append(Postprocess.denorm_images(stat_fl,input_images_,channel=c,var=args[c]))           
        input_images_denorm = np.stack(input_images_all_channles_denorm, axis=-1)
        return input_images_denorm
    
    @staticmethod
    def get_one_seq_and_time_from_batch(input_images,t_starts,i):
        assert (len(np.array(input_images).shape)==5)
        input_images_ = input_images[i,:,:,:,:]
        return input_images_

    @staticmethod
    def generate_seq_timestamps(t_start,len_seq=20):
        if isinstance(t_start,int): t_start = str(t_start)
        if isinstance(t_start,np.ndarray):t_start = str(t_start[0])
        s_datetime = datetime.datetime.strptime(t_start, '%Y%m%d%H')
        seq_ts = [s_datetime + datetime. timedelta(hours = i+1) for i in range(len_seq)]
        return seq_ts


    def save_to_netcdf_for_stochastic_generate_images(self,persistent_images_,input_images_,gen_images_stochastic,fl_name="test.nc"):
        """
        Save the input images, persistent images and generated stochatsic images to netCDF file
        args:
            input_images_        : list/array, [seq,lat,lon,channel]
            persistent_images_   : list/array, [seq,lat,lon,channel]
            gen_images_stochastic: list/array (float), [stochastic_number,seq,lat,lon,channel]
            fl_name              : str, the netcdf file name to be saved
        """
        assert (len(np.array(input_images_).shape)==len(np.array(gen_images_stochastic).shape))-1
        y_len = len(lats)
        x_len = len(lons)
        ts_len = len(self.ts)
        ts_input = self.ts[:self.context_frames]
        ts_forecast = self.ts[self.context_frames:]
        gen_images_ = np.array(gen_images_stochastic)
        output_file = os.path.join(self.output_dir,fl_name)
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
            t2[:,:,:] = self.input_images_[:self.context_frames,:,:,0]

            #mean sea level pressure
            msl = nc_file.createVariable("/analysis/inputs/MSL","f4",("time_input","lat","lon"), zlib = True)
            msl.units = 'Pa'
            msl[:,:,:] = self.input_images_[:self.context_frames,:,:,1]

            #Geopotential at 500 
            gph500 = nc_file.createVariable("/analysis/inputs/GPH500","f4",("time_input","lat","lon"), zlib = True)
            gph500.units = 'm'
            gph500[:,:,:] = self.input_images_[:self.context_frames,:,:,2]
        
            #####sub group for reference(ground truth)
            #Temperature
            t2_r = nc_file.createVariable("/analysis/reference/T2","f4",("time_forecast","lat","lon"), zlib = True)
            t2_r.units = 'K'
            t2_r[:,:,:] = self.input_images_[self.context_frames:,:,:,0]

             #mean sea level pressure
            msl_r = nc_file.createVariable("/analysis/reference/MSL","f4",("time_forecast","lat","lon"), zlib = True)
            msl_r.units = 'Pa'
            msl_r[:,:,:] = self.input_images_[self.context_frames:,:,:,1]

            #Geopotential at 500 
            gph500_r = nc_file.createVariable("/analysis/reference/GPH500","f4",("time_forecast","lat","lon"), zlib = True)
            gph500_r.units = 'm'
            gph500_r[:,:,:] = self.input_images_[self.context_frames:,:,:,2]

            ###subgroup for Pesistent analysis #######
            t2_p = nc_file.createVariable("/analysis/persistent/T2","f4",("time_forecast","lat","lon"), zlib = True)
            t2_p.units = 'K'
            t2_p[:,:,:] = self.persistent_images_[self.context_frames:,:,:,0]

            msl_p = nc_file.createVariable("/analysis/persistent/MSL","f4",("time_forecast","lat","lon"), zlib = True)
            msl_p.units = 'Pa'
            msl_p[:,:,:] = self.persistent_images_[self.context_frames:,:,:,1]
             
            #Geopotential at 500 
            gph500_p = nc_file.createVariable("/analysis/persistent/GPH500","f4",("time_forecast","lat","lon"), zlib = True)
            gph500_p.units = 'm'
            gph500_p[:,:,:] = self.persistent_images_[self.context_frames:,:,:,2]


            ################ forecast group  #####################
            for stochastic_sample_ind in self.num_stochastic_samples:
                #Temperature:
                t2 = nc_file.createVariable("/forecast/T2/stochastic/{}".format(stochastic_sample_ind),"f4",("time_forecast","lat","lon"), zlib = True)
                t2.units = 'K'
                t2[:,:,:] = gen_images_[self.context_frames-1:,:,:,0]

                #mean sea level pressure
                msl = nc_file.createVariable("/forecast/MSL/stochastic/{}".format(stochastic_sample_ind),"f4",("time_forecast","lat","lon"), zlib = True)
                msl.units = 'Pa'
                msl[:,:,:] = gen_images_[self.context_frames-1:,:,:,1]

                #Geopotential at 500 
                gph500 = nc_file.createVariable("/forecast/GPH500/stochastic/{}".format(stochastic_sample_ind),"f4",("time_forecast","lat","lon"), zlib = True)
                gph500.units = 'm'
                gph500[:,:,:] = gen_images_[self.context_frames-1:,:,:,2]        

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
        gs = gridspec.GridSpec(1, 10)
        gs.update(wspace = 0., hspace = 0.)
        xlables = [round(i,2) for i  in list(np.linspace(np.min(lons),np.max(lons),5))]
        ylabels = [round(i,2) for i  in list(np.linspace(np.max(lats),np.min(lats),5))]
        for i in range(len(ts)):
            t = ts[i]
            ax1 = plt.subplot(gs[i])
            plt.imshow(imgs[i] ,cmap = 'jet', vmin=270, vmax=300)
            ax1.title.set_text("t = " + t.strftime("%Y%m%d%H"))
            plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])
            if i == 0:
                plt.setp([ax1], xticks = list(np.linspace(0, len(lons), 5)), xticklabels = xlables, yticks = list(np.linspace(0, len(lats), 5)), yticklabels = ylabels)
                plt.ylabel(label, fontsize=10)
        plt.savefig(os.path.join(output_png_dir, label + "_TS_" + str(ts[0]) + ".jpg"))
        plt.clf()
        output_fname = label + "_TS_" + ts[0].strftime("%Y%m%d%H") + ".jpg"
        print("image {} saved".format(output_fname))

    
    @staticmethod
    def get_persistence(ts,input_dir_pkl):
        """This function gets the persistence forecast.
        'Today's weather will be like yesterday's weather.
    
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
        for t in range(len(ts)): # Scarlet: this certainly can be made nicer with list comprehension 
            ts_temp = ts[t] - datetime.timedelta(days=1)
            ts_persistence.append(ts_temp)
        t_persistence_start = ts_persistence[0]
        t_persistence_end = ts_persistence[-1]
        year_start = t_persistence_start.year
        month_start = t_persistence_start.month
        month_end = t_persistence_end.month
    
        # only one pickle file is needed (all hours during the same month)
        if month_start == month_end: 
            # Open files to search for the indizes of the corresponding time
            time_pickle  = load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'T')
            # Open file to search for the correspoding meteorological fields
            var_pickle  = load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'X')
            # Retrieve starting index
            ind = list(time_pickle).index(np.array(ts_persistence[0]))
            #print('Scarlet, Original', ts_persistence)
            #print('From Pickle', time_pickle[ind:ind+len(ts_persistence)])
        
            var_persistence  = var_pickle[ind:ind+len(ts_persistence)]
            time_persistence = time_pickle[ind:ind+len(ts_persistence)].ravel()
            print(' Scarlet Shape of time persistence',time_persistence.shape)
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
        
            # Open files to search for the indizes of the corresponding time
            time_pickle_first  = self.load_pickle_for_persistence(year_start, month_start, 'T')
            time_pickle_second = self.load_pickle_for_persistence(year_start, month_end, 'T')
        
            # Open file to search for the correspoding meteorological fields
            var_pickle_first  = self.load_pickle_for_persistence(year_start, month_start, 'X')
            var_pickle_second = self.load_pickle_for_persistence(year_start, month_end, 'X')
        
            # Retrieve starting index
            ind_first_m = list(time_pickle_first).index(np.array(t_persistence_first_m[0]))
            ind_second_m = list(time_pickle_second).index(np.array(t_persistence_second_m[0]))
        
            #print('Scarlet, Original', ts_persistence)
            #print('From Pickle', time_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)], time_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)])
            #print(' Scarlet before', time_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)].shape, time_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)].shape)
        
            # append the sequence of the second month to the first month
            var_persistence  = np.concatenate((var_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)], 
                                          var_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)]), 
                                          axis=0)
            time_persistence = np.concatenate((time_pickle_first[ind_first_m:ind_first_m+len(t_persistence_first_m)],
                                          time_pickle_second[ind_second_m:ind_second_m+len(t_persistence_second_m)]), 
                                          axis=0).ravel() # ravel is needed to eliminate the unnecessary dimension (20,1) becomes (20,)
            print(' Scarlet concatenate and ravel (time)', var_persistence.shape, time_persistence.shape)
            
            
        # tolist() is needed for plotting
        return var_persistence, time_persistence.tolist()
    
    def load_pickle_for_persistence(self,year_start, month_start, pkl_type):
        """Helper to get the content of the pickle files. There are two types in our workflow:
        T_[month].pkl where the time stamp is stored
        X_[month].pkl where the variables are stored, e.g. temperature, geopotential and pressure
        This helper function constructs the directory, opens the file to read it, returns the variable. 
        """
        path_to_pickle = self.input_dir_pkl+'/'+str(year_start)+'/'+pkl_type+'_{:02}.pkl'.format(month_start)
        infile = open(path_to_pickle,'rb')    
        var = pickle.load(infile)
        return var


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
    parser.add_argument("--dataset", type = str, help = "dataset class name")
    parser.add_argument("--dataset_hparams", type = str,
                        help = "a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type = str, help = "model class name")
    parser.add_argument("--batch_size", type = int, default = 8, help = "number of samples in batch")
    parser.add_argument("--num_samples", type = int, help = "number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type = int, default = 1)
    parser.add_argument("--num_stochastic_samples", type = int, default = 1)
    parser.add_argument("--gpu_mem_frac", type = float, default = 0.95, help = "fraction of gpu memory to use")
    parser.add_argument("--seed", type = int, default = 7)
    args = parser.parse_args()

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    test_instance = Postprocess(input_dir=args.input_dir,results_dir=args.results_dir,checkpoint=args.checkpoint,mode="test",
                      batch_size=None,num_samples=args.num_samples,num_stochastic_samples=args.num_stochastic_samples,
                      gpu_mem_frac=args.gpu_mem_frac,seed=args.seed,args=args)

    test_instance.setup()
    test_instance.run()


if __name__ == '__main__':
    main() 

