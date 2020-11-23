from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler, Michael Langguth"
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


class Postprocess(TrainModel):
    def __init__(self,input_dir=None,results_dir=None,checkpoint=None,mode="test",
                      batch_size=None,num_samples=None,num_stochastic_samples=None,
                      gpu_mem_frac=None,seed=None,args=None):
        """
        The function for inference, generate results and images
        input_dir     :str
        results_dir   :str 
        checkpoint    :str, the directory point to the checkpoints
        mode          :str, default is test, could be "train","val", and "test"
        dataset       :str, the dataset type, "era5","moving_mnist", or "kth"
        """
        #super(Postprocess,self).__init__(input_dir=input_dir,output_dir=None,datasplit_dir=data_split_dir,
        #                                  model_hparams_dict=model_hparams_dict,model=model,checkpoint=checkpoint,dataset=dataset,
        #                                  gpu_mem_frac=gpu_mem_frac,seed=seed,args=args)        
        self.results_dir  = results_dir
        self.batch_size = batch_size
        self.num_stochastic_samples = num_stochastic_samples
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.num_samplees = num_samples
        self.input_dir_tfrecords = os.path.join(self.input_dir,"tfrecords")
        self.input_dir_pkl = os.path.join(self.input_dir,"pickle") 
        if checkpoint is None: raise ("The directory point to checkpoint is empty, must be provided for postprocess step")     
        self.args = args 


    def __call__(self):
        self.set_seed()
        self.load_params_from_checkpoints_dir()
        self.setup_test_dataset()
        self.setup_model()
        self.setup_num_samples_per_epoch()
        self.make_test_dataset_iterator() 
        self.setup_graph()
        self.save_dataset_model_params_to_checkpoint_dir(dataset=self.test_dataset,video_model=self.video_model)
        self.setup_gpu_config()
        self.initia_save_data()
        self.lats,self.lons = self.get_coordinates()
        self.check_stochastic_samples_ind_based_on_model()
                

    def setup_num_samples_per_epoch(self):
        if self.num_samples:
            if num_samples > self.dataset.num_examples_per_epoch():
                raise ValueError('num_samples cannot be larger than the dataset')
            self.num_examples_per_epoch = self.num_samples
        else:
            self.num_examples_per_epoch = self.dataset.num_examples_per_epoch()
        return self.num_samples_per_epoch

 
    def setup_test_dataset(self):
        VideoDataset = datasets.get_dataset_class(self.dataset)
        self.test_dataset = VideoDataset(input_dir=self.input_dir_tfrecords,mode=self.mode,datasplit_config=self.datasplit_dict)
                            
                          
    def make_test_dataset_iterator(self):
        self.inputs = self.test_dataset.make_batch(self.batch_size)
        self.inputs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in self.inputs.items()}
        

    def get_coordinates(self):
        """
        Retrieves the latitudes and longitudes read from the metadata json file.
        """
        metadata_fname = os.path.join(self.input_dir,"metadata.json")
        md = MetaData(json_file=metadata_fname)
        md.get_metadata_from_file(metadata_fname)
    
        try:
            print("lat:",md.lat)
            print("lon:",md.lon)
            return md.lat, md.lon
        except:
            raise ValueError("Error when handling: '"+metadata_fname+"'")
    
    def initia_save_data(self):
        self.sample_ind = 0
        self.gen_images_all = []
        self.persistent_images_all = []
        self.input_images_all = []
        return self.sample_ind, self.gen_images_all, self.persistent_images_all, self.input_images_all


    def check_stochastic_samples_ind_based_on_model(self):
        if self.model == "convLSTM": self.num_stochastic_samples = 1

    
    def init_session(self):
        self.sess = tf.Session(config = self.config)
        self.sess.graph.as_default()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def run_inputs_per_batch(self):
        self.input_results = sess.run(inputs)
        self.input_images = self.input_results["images"]
        #get one seq and the corresponding start time poin
        self.t_starts = self.input_results["T_start"]
        self.input_images_,self.t_start = get_one_seq_and_time(self.input_images,self.t_starts,i)
        #Renormalized data for inputs
        stat_fl = os.path.join(args.input_dir,"pickle/statistics.json")
        self.input_images_denorm = denorm_images_all_channels(stat_fl,self.input_images_,["T2","MSL","gph500"])



    def run_test(self):
        self.init_session()     
        self.restore(self.sess, args.checkpoint)
        #Loop for samples
        while self.sample_ind < self.num_samples_per_epoch:
            gen_images_stochastic = []
            if self.num_samples_per_epoch < self.sample_ind:
                break
            else:
               self.run_inputs_per_batch()

            #Loop for stochastics 
            feed_dict = {input_ph: input_results[name] for name, input_ph in self.inputs.items()}
            for stochastic_sample_ind in range(self.num_stochastic_samples):
                gen_images = sess.run(model.outputs['gen_images'], feed_dict = feed_dict)#return [batchsize,seq_len,lat,lon,channel]
                assert gen_images.shape[1] == sequence_length-1 #The generate images seq_len should be sequence_len -1, since the last one is not used for comparing with groud truth 
                for i in range(args.batch_size):
                    #generate time stamps for sequences
                    ts = generate_seq_timestamps(self.t_start,len_seq=sequence_length)
                    #Renormalized data for generate
                    gen_images_ = gen_images[i]
                    gen_images_denorm = denorm_images_all_channels(stat_fl,gen_images_,["T2","MSL","gph500"])
                    
                    #Save input to netCDF file for each stochastic sample
                    init_date_str = ts[0].strftime("%Y%m%d%H")
                    save_to_netcdf_per_sequence(self.results_dir,self.input_images_denorm,gen_images_denorm,self.lons,self.lats,ts,self.context_frames,self.future_length,args.model,fl_name="vfp_{}.nc".format(init_date_str))
                                                             
                    #Generate images inputs
                    plot_seq_imgs(imgs=input_images_denorm[context_frames+1:,:,:,0],lats=lats,lons=lons,ts=ts[context_frames+1:],label="Ground Truth",output_png_dir=args.results_dir)  
                                                             
                    #Generate forecast images
                    plot_seq_imgs(imgs=gen_images_denorm[context_frames:,:,:,0],lats=lats,lons=lons,ts=ts[context_frames+1:],label="Forecast by Model " + args.model,output_png_dir=args.results_dir) 
            
          
                    persistence_images, ts_persistence = get_persistence(ts, input_dir_pkl)
                    # I am not sure about the number of frames given with context_frames and context_frames +1
                    plot_seq_imgs(imgs=persistence_images[context_frames+1:,:,:,0],lats=lats,lons=lons,ts=ts_persistence[context_frames+1:], 
                          label="Persistence Forecast" + args.model,output_png_dir=args.results_dir)

            sample_ind += args.batch_size




    @staticmethod
    def setup_dirs(input_dir,results_png_dir):
        input_dir = args.input_dir
        temporal_dir = os.path.split(input_dir)[0] + "/hickle/splits/"
        print ("temporal_dir:",temporal_dir)


    @staticmethod
    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    @staticmethod
    def denorm_images(stat_fl, input_images_,channel,var):
        norm_cls  = Norm_data(var)
        norm = 'minmax'
        with open(stat_fl) as js_file:
             norm_cls.check_and_set_norm(json.load(js_file),norm)
        input_images_denorm = norm_cls.denorm_var(input_images_[:, :, :,channel], var, norm)
        return input_images_denorm

    def denorm_images_all_channels(stat_fl,input_images_,*args):
        input_images_all_channles_denorm = []
        input_images_ = np.array(input_images_)
        args = [item for item in args][0]
        for c in range(len(args)):
            print("args c:", args[c])
            input_images_all_channles_denorm.append(denorm_images(stat_fl,input_images_,channel=c,var=args[c]))           
        input_images_denorm = np.stack(input_images_all_channles_denorm, axis=-1)
        return input_images_denorm

    def get_one_seq_and_time(input_images,t_starts,i):
        assert (len(np.array(input_images).shape)==5)
        input_images_ = input_images[i,:,:,:,:]
        t_start = t_starts[i]
        return input_images_,t_start

    def generate_seq_timestamps(t_start,len_seq=20):
        if isinstance(t_start,int): t_start = str(t_start)
        if isinstance(t_start,np.ndarray):t_start = str(t_start[0])
        s_datetime = datetime.datetime.strptime(t_start, '%Y%m%d%H')
        seq_ts = [s_datetime + datetime. timedelta(hours = i+1) for i in range(len_seq)]
        return seq_ts
    
    
    def save_to_netcdf_per_sequence(output_dir,input_images_,gen_images_,lons,lats,ts,context_frames,future_length,model_name,fl_name="test.nc"):
        assert (len(np.array(input_images_).shape)==len(np.array(gen_images_).shape))
        y_len = len(lats)
        x_len = len(lons)
        ts_len = len(ts)
        ts_input = ts[:context_frames]
        ts_forecast = ts[context_frames:]
        gen_images_ = np.array(gen_images_)
        output_file = os.path.join(output_dir,fl_name)
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
            timeD = nc_file.createDimension('time_input', context_frames) 
            timeF = nc_file.createDimension('time_forecast', future_length)

            #Latitude
            lat  = nc_file.createVariable('lat', float, ('lat',), zlib = True)
            lat.units = 'degrees_north'
            lat[:] = lats


            #Longitude
            lon = nc_file.createVariable('lon', float, ('lon',), zlib = True)
            lon.units = 'degrees_east'
            lon[:] = lons

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
            t2[:,:,:] = input_images_[:context_frames,:,:,0]

            #mean sea level pressure
            msl = nc_file.createVariable("/analysis/inputs/MSL","f4",("time_input","lat","lon"), zlib = True)
            msl.units = 'Pa'
            msl[:,:,:] = input_images_[:context_frames,:,:,1]

            #Geopotential at 500 
            gph500 = nc_file.createVariable("/analysis/inputs/GPH500","f4",("time_input","lat","lon"), zlib = True)
            gph500.units = 'm'
            gph500[:,:,:] = input_images_[:context_frames,:,:,2]
        
            #####sub group for reference(ground truth)
            #Temperature
            t2_r = nc_file.createVariable("/analysis/reference/T2","f4",("time_forecast","lat","lon"), zlib = True)
            t2_r.units = 'K'
            t2_r[:,:,:] = input_images_[context_frames:,:,:,0]

             #mean sea level pressure
            msl_r = nc_file.createVariable("/analysis/reference/MSL","f4",("time_forecast","lat","lon"), zlib = True)
            msl_r.units = 'Pa'
            msl_r[:,:,:] = input_images_[context_frames:,:,:,1]

            #Geopotential at 500 
            gph500_r = nc_file.createVariable("/analysis/reference/GPH500","f4",("time_forecast","lat","lon"), zlib = True)
            gph500_r.units = 'm'
            gph500_r[:,:,:] = input_images_[context_frames:,:,:,2]
        

            ################ forecast group  #####################
            #Temperature:
            t2 = nc_file.createVariable("/forecast/{}/T2".format(model_name),"f4",("time_forecast","lat","lon"), zlib = True)
            t2.units = 'K'
            print ("gen_images_ 20200822:",np.array(gen_images_).shape)
            t2[:,:,:] = gen_images_[context_frames-1:,:,:,0]
            print("NetCDF created")

            #mean sea level pressure
            msl = nc_file.createVariable("/forecast/{}/MSL".format(model_name),"f4",("time_forecast","lat","lon"), zlib = True)
            msl.units = 'Pa'
            msl[:,:,:] = gen_images_[context_frames-1:,:,:,1]

            #Geopotential at 500 
            gph500 = nc_file.createVariable("/forecast/{}/GPH500".format(model_name),"f4",("time_forecast","lat","lon"), zlib = True)
            gph500.units = 'm'
            gph500[:,:,:] = gen_images_[context_frames-1:,:,:,2]        

            print("{} created".format(output_file)) 

        return None

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

    
def get_persistence(ts, input_dir_pkl):
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
        time_pickle_first  = load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'T')
        time_pickle_second = load_pickle_for_persistence(input_dir_pkl, year_start, month_end, 'T')
        
        # Open file to search for the correspoding meteorological fields
        var_pickle_first  = load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'X')
        var_pickle_second = load_pickle_for_persistence(input_dir_pkl, year_start, month_end, 'X')
        
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

    
    
def load_pickle_for_persistence(input_dir_pkl, year_start, month_start, pkl_type):
    """Helper to get the content of the pickle files. There are two types in our workflow:
    T_[month].pkl where the time stamp is stored
    X_[month].pkl where the variables are stored, e.g. temperature, geopotential and pressure
    This helper function constructs the directory, opens the file to read it, returns the variable. 
    """
    path_to_pickle = input_dir_pkl+'/'+str(year_start)+'/'+pkl_type+'_{:02}.pkl'.format(month_start)
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
    parser.add_argument("--mode", type = str, choices = ['train','val', 'test'], default = 'val',
                        help = 'mode for dataset, val or test.')
    parser.add_argument("--dataset", type = str, help = "dataset class name")
    parser.add_argument("--dataset_hparams", type = str,
                        help = "a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type = str, help = "model class name")
    parser.add_argument("--model_hparams", type = str,
                        help = "a string of comma separated list of model hyperparameters")
    parser.add_argument("--batch_size", type = int, default = 8, help = "number of samples in batch")
    parser.add_argument("--num_samples", type = int, help = "number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type = int, default = 1)
    parser.add_argument("--num_stochastic_samples", type = int, default = 1)
    parser.add_argument("--gif_length", type = int, help = "default is sequence_length")
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
     

    while sample_ind < 5:
        gen_images_stochastic = []
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results = sess.run(inputs)
            input_images = input_results["images"]
            #get the intial times
            t_starts = input_results["T_start"]
        except tf.errors.OutOfRangeError:
            break
            
        #Get prediction values 
        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        #loop for each stochastic sample
        for stochastic_sample_ind in range(args.num_stochastic_samples):

if __name__ == '__main__':
    main() 

