from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import math
import random
import cv2
import numpy as np
import tensorflow as tf
import pickle
from random import seed
import random
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import pandas as pd
import re
from video_prediction import datasets, models
from matplotlib.colors import LinearSegmentedColormap
#from matplotlib.ticker import MaxNLocator
#from video_prediction.utils.ffmpeg_gif import save_gif
from skimage.metrics import structural_similarity as ssim
import datetime
# Scarlet 2020/05/28: access to statistical values in json file 
from os import path
import sys
sys.path.append(path.abspath('../video_prediction/datasets/'))
from era5_dataset_v2 import Norm_data
from os.path import dirname
from netCDF4 import Dataset,date2num
from metadata import MetaData as MetaData

def set_seed(seed):
    if seed is not None:
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed) 

def get_coordinates(metadata_fname):
    """
    Retrieves the latitudes and longitudes read from the metadata json file.
    """
    md = MetaData(json_file=metadata_fname)
    md.get_metadata_from_file(metadata_fname)
    
    try:
        print("lat:",md.lat)
        print("lon:",md.lon)
        return md.lat, md.lon
    except:
        raise ValueError("Error when handling: '"+metadata_fname+"'")
    

def load_checkpoints_and_create_output_dirs(checkpoint,dataset,model):
    if checkpoint:
        checkpoint_dir = os.path.normpath(checkpoint)
        if not os.path.isdir(checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % checkpoint)
            options = json.loads(f.read())
            dataset = dataset or options['dataset']
            model = model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
    else:
        if not dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not model:
            raise ValueError('model is required when checkpoint is not specified')

    return options,dataset,model, checkpoint_dir,dataset_hparams_dict,model_hparams_dict


    
def setup_dataset(dataset,input_dir,mode,seed,num_epochs,dataset_hparams,dataset_hparams_dict):
    VideoDataset = datasets.get_dataset_class(dataset)
    dataset = VideoDataset(
        input_dir,
        mode = mode,
        num_epochs = num_epochs,
        seed = seed,
        hparams_dict = dataset_hparams_dict,
        hparams = dataset_hparams)
    return dataset


def setup_dirs(input_dir,results_png_dir):
    input_dir = args.input_dir
    temporal_dir = os.path.split(input_dir)[0] + "/hickle/splits/"
    print ("temporal_dir:",temporal_dir)


def update_hparams_dict(model_hparams_dict,dataset):
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': dataset.hparams.context_frames,
        'sequence_length': dataset.hparams.sequence_length,
        'repeat': dataset.hparams.time_shift,
    })
    return hparams_dict


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def setup_num_samples_per_epoch(num_samples, dataset):
    if num_samples:
        if num_samples > dataset.num_examples_per_epoch():
            raise ValueError('num_samples cannot be larger than the dataset')
        num_examples_per_epoch = num_samples
    else:
        num_examples_per_epoch = dataset.num_examples_per_epoch()
    #if num_examples_per_epoch % args.batch_size != 0:
    #    raise ValueError('batch_size should evenly divide the dataset size %d' % num_examples_per_epoch)
    return num_examples_per_epoch


def initia_save_data():
    sample_ind = 0
    gen_images_all = []
    #Bing:20200410
    persistent_images_all = []
    input_images_all = []
    return sample_ind, gen_images_all,persistent_images_all, input_images_all


def write_params_to_results_dir(args,output_dir,dataset,model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys = True, indent = 4))
    with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(dataset.hparams.values(), sort_keys = True, indent = 4))
    with open(os.path.join(output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys = True, indent = 4))
    return None

def get_one_seq_and_time(input_images,i):
    assert (len(np.array(input_images).shape)==5)
    input_images_ = input_images[i,:,:,:,:]
    return input_images_


def denorm_images_all_channels(input_images_):
    input_images_all_channles_denorm = []
    input_images_ = np.array(input_images_)
    input_images_denorm = input_images_ * 255.0
    #print("input_images_denorm shape",input_images_denorm.shape)
    return input_images_denorm

def plot_seq_imgs(imgs,output_png_dir,idx,label="Ground Truth"):
    """
    Plot the seq images 
    """

    if len(np.array(imgs).shape)!=3:raise("img dims should be three: (seq_len,lat,lon)")
    img_len = imgs.shape[0]
    fig = plt.figure(figsize=(18,6))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace = 0., hspace = 0.)
    for i in range(img_len):      
        ax1 = plt.subplot(gs[i])
        plt.imshow(imgs[i] ,cmap = 'jet')
        plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])
    plt.savefig(os.path.join(output_png_dir, label + "_" +   str(idx) +  ".jpg"))
    print("images_saved")
    plt.clf()
 

    
def get_persistence(ts):
    pass


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
    parser.add_argument("--fps", type = int, default = 4)
    parser.add_argument("--gpu_mem_frac", type = float, default = 0.95, help = "fraction of gpu memory to use")
    parser.add_argument("--seed", type = int, default = 7)
    args = parser.parse_args()
    set_seed(args.seed)

    dataset_hparams_dict = {}
    model_hparams_dict = {}

    options,dataset,model, checkpoint_dir,dataset_hparams_dict,model_hparams_dict = load_checkpoints_and_create_output_dirs(args.checkpoint,args.dataset,args.model)
    print("Step 1 finished")

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    #setup dataset and model object
    input_dir_tf = os.path.join(args.input_dir, "tfrecords") # where tensorflow records are stored
    dataset = setup_dataset(dataset,input_dir_tf,args.mode,args.seed,args.num_epochs,args.dataset_hparams,dataset_hparams_dict)
    
    print("Step 2 finished")
    VideoPredictionModel = models.get_model_class(model)
    
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': dataset.hparams.context_frames,
        'sequence_length': dataset.hparams.sequence_length,
        'repeat': dataset.hparams.time_shift,
    })
    
    model = VideoPredictionModel(
        mode = args.mode,
        hparams_dict = hparams_dict,
        hparams = args.model_hparams)

    sequence_length = model.hparams.sequence_length
    context_frames = model.hparams.context_frames
    future_length = sequence_length - context_frames #context_Frames is the number of input frames

    num_examples_per_epoch = setup_num_samples_per_epoch(args.num_samples,dataset)
    
    inputs = dataset.make_batch(args.batch_size)
    print("inputs",inputs)
    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
    print("input_phs",input_phs)
    
    
    # Build graph
    with tf.variable_scope(''):
        model.build_graph(input_phs)

    #Write the update hparameters into results_dir    
    write_params_to_results_dir(args=args,output_dir=args.results_dir,dataset=dataset,model=model)
        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
    sess = tf.Session(config = config)
    sess.graph.as_default()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.restore(sess, args.checkpoint)
    
    #model.restore(sess, args.checkpoint)#Bing: Todo: 20200728 Let's only focus on true and persistend data
    sample_ind, gen_images_all, persistent_images_all, input_images_all = initia_save_data()
    
    is_first=True
    #loop for in samples
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
        gen_images = sess.run(model.outputs['gen_images'], feed_dict = feed_dict)#return [batchsize,seq_len,lat,lon,channel]
        print("gen_images 20200822:",np.array(gen_images).shape)       
        #Loop in batch size
        for i in range(args.batch_size):
            
            #get one seq and the corresponding start time point
            input_images_ = get_one_seq_and_time(input_images,i)
            
            #Renormalized data for inputs
            input_images_denorm = denorm_images_all_channels(input_images_)  
            print("input_images_denorm",input_images_denorm[0][0])
                                                             
            #Renormalized data for inputs
            gen_images_ = gen_images[i]
            gen_images_denorm = denorm_images_all_channels(gen_images_)
            print("gene_images_denorm:",gen_images_denorm[0][0])
            
            #Generate images inputs
            plot_seq_imgs(imgs=input_images_denorm[:context_frames-1,:,:,0],idx = sample_ind + i, label="Ground Truth",output_png_dir=args.results_dir)  
                                                             
            #Generate forecast images
            plot_seq_imgs(imgs=gen_images_denorm[context_frames:,:,:,0],idx = sample_ind + i,label="Forecast by Model " + args.model,output_png_dir=args.results_dir) 
            
            #TODO: Scaret plot persistence image
            #implment get_persistence() function

            #in case of generate the images for all the input, we just generate the first 5 sampe_ind examples for visuliation

        sample_ind += args.batch_size


        #for input_image in input_images_:

#             for stochastic_sample_ind in range(args.num_stochastic_samples):
#                 input_images_all.extend(input_images)
#                 with open(os.path.join(args.output_png_dir, "input_images_all.pkl"), "wb") as input_files:
#                     pickle.dump(list(input_images_all), input_files)


#                 gen_images_stochastic.append(gen_images)
#                 #print("Stochastic_sample,", stochastic_sample_ind)
#                 for i in range(args.batch_size):
#                     #bing:20200417
#                     t_stampe = test_temporal_pkl[sample_ind+i]
#                     print("timestamp:",type(t_stampe))
#                     persistent_ts = np.array(t_stampe) - datetime.timedelta(days=1)
#                     print ("persistent ts",persistent_ts)
#                     persistent_idx = list(test_temporal_pkl).index(np.array(persistent_ts))
#                     persistent_X = X_test[persistent_idx:persistent_idx+context_frames + future_length]
#                     print("persistent index in test set:", persistent_idx)
#                     print("persistent_X.shape",persistent_X.shape)
#                     persistent_images_all.append(persistent_X)

#                     cmap_name = 'my_list'
#                     if sample_ind < 100:
#                         #name = '_Stochastic_id_' + str(stochastic_sample_ind) + 'Batch_id_' + str(
#                         #    sample_ind) + " + Sample_" + str(i)
#                         name = '_Stochastic_id_' + str(stochastic_sample_ind) + "_Time_"+ t_stampe[0].strftime("%Y%m%d-%H%M%S")
#                         print ("name",name)
#                         gen_images_ = np.array(list(input_images[i,:context_frames]) + list(gen_images[i,-future_length:, :]))
#                         #gen_images_ =  gen_images[i, :]
#                         input_images_ = input_images[i, :]
#                         #Bing:20200417
#                         #persistent_images = ?
#                         #+++Scarlet:20200528   
#                         #print('Scarlet1')
#                         input_gen_diff = norm_cls.denorm_var(input_images_[:, :, :,0], 'T2', norm) - norm_cls.denorm_var(gen_images_[:, :, :, 0],'T2',norm)
#                         persistent_diff = norm_cls.denorm_var(input_images_[:, :, :,0], 'T2', norm) - norm_cls.denorm_var(persistent_X[:, :, :, 0], 'T2',norm)
#                         #---Scarlet:20200528    
#                         gen_mse_avg_ = [np.mean(input_gen_diff[frame, :, :] ** 2) for frame in
#                                         range(sequence_length)]  # return the list with 10 (sequence) mse
#                         persistent_mse_avg_ = [np.mean(persistent_diff[frame, :, :] ** 2) for frame in
#                                         range(sequence_length)]  # return the list with 10 (sequence) mse

#                         fig = plt.figure(figsize=(18,6))
#                         gs = gridspec.GridSpec(1, 10)
#                         gs.update(wspace = 0., hspace = 0.)
#                         ts = list(range(10,20)) #[10,11,12,..]
#                         xlables = [round(i,2) for i  in list(np.linspace(np.min(lon),np.max(lon),5))]
#                         ylabels = [round(i,2) for i  in list(np.linspace(np.max(lat),np.min(lat),5))]

#                         for t in ts:

#                             #if t==0 : ax1=plt.subplot(gs[t])
#                             ax1 = plt.subplot(gs[ts.index(t)])
#                             #+++Scarlet:20200528
#                             #print('Scarlet2')
#                             input_image = norm_cls.denorm_var(input_images_[t, :, :, 0], 'T2', norm)
#                             #---Scarlet:20200528
#                             plt.imshow(input_image, cmap = 'jet', vmin=270, vmax=300)
#                             ax1.title.set_text("t = " + str(t+1-10))
#                             plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])
#                             if t == 0:
#                                 plt.setp([ax1], xticks = list(np.linspace(0, 64, 3)), xticklabels = xlables, yticks = list(np.linspace(0, 64, 3)), yticklabels = ylabels)
#                                 plt.ylabel("Ground Truth", fontsize=10)
#                         plt.savefig(os.path.join(args.output_png_dir, "Ground_Truth_Sample_" + str(name) + ".jpg"))
#                         plt.clf()

#                         fig = plt.figure(figsize=(12,6))
#                         gs = gridspec.GridSpec(1, 10)
#                         gs.update(wspace = 0., hspace = 0.)

#                         for t in ts:
#                             #if t==0 : ax1=plt.subplot(gs[t])
#                             ax1 = plt.subplot(gs[ts.index(t)])
#                             #+++Scarlet:20200528
#                             #print('Scarlet3')
#                             gen_image = norm_cls.denorm_var(gen_images_[t, :, :, 0], 'T2', norm)
#                             #---Scarlet:20200528
#                             plt.imshow(gen_image, cmap = 'jet', vmin=270, vmax=300)
#                             ax1.title.set_text("t = " + str(t+1-10))
#                             plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])

#                         plt.savefig(os.path.join(args.output_png_dir, "Predicted_Sample_" + str(name) + ".jpg"))
#                         plt.clf()


#                         fig = plt.figure(figsize=(12,6))
#                         gs = gridspec.GridSpec(1, 10)
#                         gs.update(wspace = 0., hspace = 0.)
#                         for t in ts:
#                             #if t==0 : ax1=plt.subplot(gs[t])
#                             ax1 = plt.subplot(gs[ts.index(t)])
#                             #persistent_image = persistent_X[t, :, :, 0] * (321.46630859375 - 235.2141571044922) + 235.2141571044922
#                             plt.imshow(persistent_X[t, :, :, 0], cmap = 'jet', vmin=270, vmax=300)
#                             ax1.title.set_text("t = " + str(t+1-10))
#                             plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])

#                         plt.savefig(os.path.join(args.output_png_dir, "Persistent_Sample_" + str(name) + ".jpg"))
#                         plt.clf()

                        
#                 with open(os.path.join(args.output_png_dir, "persistent_images_all.pkl"), "wb") as input_files:
#                     pickle.dump(list(persistent_images_all), input_files)
#                     print ("Save persistent all")
#                 if is_first:
#                     gen_images_all = gen_images_stochastic
#                     is_first = False
#                 else:
#                     gen_images_all = np.concatenate((np.array(gen_images_all), np.array(gen_images_stochastic)), axis=1)

#                 if args.num_stochastic_samples == 1:
#                     with open(os.path.join(args.output_png_dir, "gen_images_all.pkl"), "wb") as gen_files:
#                         pickle.dump(list(gen_images_all[0]), gen_files)
#                         print ("Save generate all")
#                 else:
#                     with open(os.path.join(args.output_png_dir, "gen_images_sample_id_" + str(sample_ind)),"wb") as gen_files:
#                         pickle.dump(list(gen_images_stochastic), gen_files)
#                     with open(os.path.join(args.output_png_dir, "gen_images_all_stochastic"), "wb") as gen_files:
#                         pickle.dump(list(gen_images_all), gen_files)

#         sample_ind += args.batch_size


#     with open(os.path.join(args.output_png_dir, "input_images_all.pkl"),"rb") as input_files:
#         input_images_all = pickle.load(input_files)

#     with open(os.path.join(args.output_png_dir, "gen_images_all.pkl"),"rb") as gen_files:
#         gen_images_all = pickle.load(gen_files)

#     with open(os.path.join(args.output_png_dir, "persistent_images_all.pkl"),"rb") as gen_files:
#         persistent_images_all = pickle.load(gen_files)

#     #+++Scarlet:20200528
#     #print('Scarlet4')
#     input_images_all = np.array(input_images_all)
#     input_images_all = norm_cls.denorm_var(input_images_all, 'T2', norm)
#     #---Scarlet:20200528
#     persistent_images_all = np.array(persistent_images_all)
#     if len(np.array(gen_images_all).shape) == 6:
#         for i in range(len(gen_images_all)):
#             #+++Scarlet:20200528
#             #print('Scarlet5')
#             gen_images_all_stochastic = np.array(gen_images_all)[i,:,:,:,:,:]
#             gen_images_all_stochastic = norm_cls.denorm_var(gen_images_all_stochastic, 'T2', norm)
#             #gen_images_all_stochastic = np.array(gen_images_all_stochastic) * (321.46630859375 - 235.2141571044922) + 235.2141571044922
#             #---Scarlet:20200528
#             mse_all = []
#             psnr_all = []
#             ssim_all = []
#             f = open(os.path.join(args.output_png_dir, 'prediction_scores_4prediction_stochastic_{}.txt'.format(i)), 'w')
#             for i in range(future_length):
#                 mse_model = np.mean((input_images_all[:, i + 10, :, :, 0] - gen_images_all_stochastic[:, i + 9, :, :,
#                                                                             0]) ** 2)  # look at all timesteps except the first
#                 psnr_model = psnr(input_images_all[:, i + 10, :, :, 0], gen_images_all_stochastic[:, i + 9, :, :, 0])
#                 ssim_model = ssim(input_images_all[:, i + 10, :, :, 0], gen_images_all_stochastic[:, i + 9, :, :, 0],
#                                   data_range = max(gen_images_all_stochastic[:, i + 9, :, :, 0].flatten()) - min(
#                                       input_images_all[:, i + 10, :, :, 0].flatten()))
#                 mse_all.extend([mse_model])
#                 psnr_all.extend([psnr_model])
#                 ssim_all.extend([ssim_model])
#                 results = {"mse": mse_all, "psnr": psnr_all, "ssim": ssim_all}
#                 f.write("##########Predicted Frame {}\n".format(str(i + 1)))
#                 f.write("Model MSE: %f\n" % mse_model)
#                 # f.write("Previous Frame MSE: %f\n" % mse_prev)
#                 f.write("Model PSNR: %f\n" % psnr_model)
#                 f.write("Model SSIM: %f\n" % ssim_model)


#             pickle.dump(results, open(os.path.join(args.output_png_dir, "results_stochastic_{}.pkl".format(i)), "wb"))
#             # f.write("Previous frame PSNR: %f\n" % psnr_prev)
#             f.write("Shape of X_test: " + str(input_images_all.shape))
#             f.write("")
#             f.write("Shape of X_hat: " + str(gen_images_all_stochastic.shape))

#     else:
#         #+++Scarlet:20200528
#         #print('Scarlet6')
#         gen_images_all = np.array(gen_images_all)
#         gen_images_all = norm_cls.denorm_var(gen_images_all, 'T2', norm)
#         #---Scarlet:20200528
        
#         # mse_model = np.mean((input_images_all[:, 1:,:,:,0] - gen_images_all[:, 1:,:,:,0])**2)  # look at all timesteps except the first
#         # mse_model_last = np.mean((input_images_all[:, future_length-1,:,:,0] - gen_images_all[:, future_length-1,:,:,0])**2)
#         # mse_prev = np.mean((input_images_all[:, :-1,:,:,0] - gen_images_all[:, 1:,:,:,0])**2 )
#         mse_all = []
#         psnr_all = []
#         ssim_all = []
#         persistent_mse_all = []
#         persistent_psnr_all = []
#         persistent_ssim_all = []
#         f = open(os.path.join(args.output_png_dir, 'prediction_scores_4prediction.txt'), 'w')
#         for i in range(future_length):
#             mse_model = np.mean((input_images_all[:1268, i + 10, :, :, 0] - gen_images_all[:, i + 9, :, :,
#                                                                         0]) ** 2)  # look at all timesteps except the first
#             persistent_mse_model = np.mean((input_images_all[:1268, i + 10, :, :, 0] - persistent_images_all[:, i + 9, :, :,
#                                                                         0]) ** 2)  # look at all timesteps except the first
            
#             psnr_model = psnr(input_images_all[:1268, i + 10, :, :, 0], gen_images_all[:, i + 9, :, :, 0])
#             ssim_model = ssim(input_images_all[:1268, i + 10, :, :, 0], gen_images_all[:, i + 9, :, :, 0],
#                               data_range = max(gen_images_all[:, i + 9, :, :, 0].flatten()) - min(
#                                   input_images_all[:, i + 10, :, :, 0].flatten()))
#             persistent_psnr_model = psnr(input_images_all[:1268, i + 10, :, :, 0], persistent_images_all[:, i + 9, :, :, 0])
#             persistent_ssim_model = ssim(input_images_all[:1268, i + 10, :, :, 0], persistent_images_all[:, i + 9, :, :, 0],
#                               data_range = max(gen_images_all[:1268, i + 9, :, :, 0].flatten()) - min(input_images_all[:1268, i + 10, :, :, 0].flatten()))
#             mse_all.extend([mse_model])
#             psnr_all.extend([psnr_model])
#             ssim_all.extend([ssim_model])
#             persistent_mse_all.extend([persistent_mse_model])
#             persistent_psnr_all.extend([persistent_psnr_model])
#             persistent_ssim_all.extend([persistent_ssim_model])
#             results = {"mse": mse_all, "psnr": psnr_all, "ssim": ssim_all}

#             persistent_results = {"mse": persistent_mse_all, "psnr": persistent_psnr_all, "ssim": persistent_ssim_all}
#             f.write("##########Predicted Frame {}\n".format(str(i + 1)))
#             f.write("Model MSE: %f\n" % mse_model)
#             # f.write("Previous Frame MSE: %f\n" % mse_prev)
#             f.write("Model PSNR: %f\n" % psnr_model)
#             f.write("Model SSIM: %f\n" % ssim_model)

#         pickle.dump(results, open(os.path.join(args.output_png_dir, "results.pkl"), "wb"))
#         pickle.dump(persistent_results, open(os.path.join(args.output_png_dir, "persistent_results.pkl"), "wb"))
#         # f.write("Previous frame PSNR: %f\n" % psnr_prev)
#         f.write("Shape of X_test: " + str(input_images_all.shape))
#         f.write("")
#         f.write("Shape of X_hat: " + str(gen_images_all.shape)      

if __name__ == '__main__':
    main()        

    #psnr_model = psnr(input_images_all[:, :10, :, :, 0],  gen_images_all[:, :10, :, :, 0])
    #psnr_model_last = psnr(input_images_all[:, 10, :, :, 0],  gen_images_all[:,10, :, :, 0])
    #psnr_prev = psnr(input_images_all[:, :, :, :, 0],  input_images_all[:, 1:10, :, :, 0])

    # ims = []
    # fig = plt.figure()
    # for frame in range(20):
    #     input_gen_diff = np.mean((np.array(gen_images_all) - np.array(input_images_all))**2, axis=0)[frame, :,:,0] # Get the first prediction frame (batch,height, width, channel)
    #     #pix_mean = np.mean(input_gen_diff, axis = 0)
    #     #pix_std = np.std(input_gen_diff, axis=0)
    #     im = plt.imshow(input_gen_diff, interpolation = 'none',cmap='PuBu')
    #     if frame == 0:
    #         fig.colorbar(im)
    #     ttl = plt.text(1.5, 2, "Frame_" + str(frame +1))
    #     ims.append([im, ttl])
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, blit = True, repeat_delay=2000)
    # ani.save(os.path.join(args.output_png_dir, "Mean_Frames.mp4"))
    # plt.close("all")

    # ims = []
    # fig = plt.figure()
    # for frame in range(19):
    #     pix_std= np.std((np.array(gen_images_all) - np.array(input_images_all))**2, axis = 0)[frame, :,:, 0]  # Get the first prediction frame (batch,height, width, channel)
    #     #pix_mean = np.mean(input_gen_diff, axis = 0)
    #     #pix_std = np.std(input_gen_diff, axis=0)
    #     im = plt.imshow(pix_std, interpolation = 'none',cmap='PuBu')
    #     if frame == 0:
    #         fig.colorbar(im)
    #     ttl = plt.text(1.5, 2, "Frame_" + str(frame+1))
    #     ims.append([im, ttl])
    # ani = animation.ArtistAnimation(fig, ims, interval = 1000, blit = True, repeat_delay = 2000)
    # ani.save(os.path.join(args.output_png_dir, "Std_Frames.mp4"))

    # seed(1)
    # s = random.sample(range(len(gen_images_all)), 100)
    # print("******KDP******")
    # #kernel density plot for checking the model collapse
    # fig = plt.figure()
    # kdp = sns.kdeplot(gen_images_all[s].flatten(), shade=True, color="r", label = "Generate Images")
    # kdp = sns.kdeplot(input_images_all[s].flatten(), shade=True, color="b", label = "Ground True")
    # kdp.set(xlabel = 'Temperature (K)', ylabel = 'Probability')
    # plt.savefig(os.path.join(args.output_png_dir, "kdp_gen_images.png"), dpi = 400)
    # plt.clf()

    #line plot for evaluating the prediction and groud-truth
    # for i in [0,3,6,9,12,15,18]:
    #     fig = plt.figure()
    #     plt.scatter(gen_images_all[:,i,:,:][s].flatten(),input_images_all[:,i,:,:][s].flatten(),s=0.3)
    #     #plt.scatter(gen_images_all[:,0,:,:].flatten(),input_images_all[:,0,:,:].flatten(),s=0.3)
    #     plt.xlabel("Prediction")
    #     plt.ylabel("Real values")
    #     plt.title("Frame_{}".format(i+1))
    #     plt.plot([250,300], [250,300],color="black")
    #     plt.savefig(os.path.join(args.output_png_dir,"pred_real_frame_{}.png".format(str(i))))
    #     plt.clf()
    #
    # mse_model_by_frames = np.mean((input_images_all[:, :, :, :, 0][s] - gen_images_all[:, :, :, :, 0][s]) ** 2,axis=(2,3)) #return (batch, sequence)
    # x = [str(i+1) for i in list(range(19))]
    # fig,axis = plt.subplots()
    # mean_f = np.mean(mse_model_by_frames, axis = 0)
    # median = np.median(mse_model_by_frames, axis=0)
    # q_low = np.quantile(mse_model_by_frames, q=0.25, axis=0)
    # q_high = np.quantile(mse_model_by_frames, q=0.75, axis=0)
    # d_low = np.quantile(mse_model_by_frames,q=0.1, axis=0)
    # d_high = np.quantile(mse_model_by_frames, q=0.9, axis=0)
    # plt.fill_between(x, d_high, d_low, color="ghostwhite",label="interdecile range")
    # plt.fill_between(x,q_high, q_low , color = "lightgray", label="interquartile range")
    # plt.plot(x, median, color="grey", linewidth=0.6, label="Median")
    # plt.plot(x, mean_f, color="peachpuff",linewidth=1.5, label="Mean")
    # plt.title(f'MSE percentile')
    # plt.xlabel("Frames")
    # plt.legend(loc=2, fontsize=8)
    # plt.savefig(os.path.join(args.output_png_dir,"mse_percentiles.png"))


##                
##
##                    # fig = plt.figure()
##                    # gs = gridspec.GridSpec(4,6)
##                    # gs.update(wspace = 0.7,hspace=0.8)
##                    # ax1 = plt.subplot(gs[0:2,0:3])
##                    # ax2 = plt.subplot(gs[0:2,3:],sharey=ax1)
##                    # ax3 = plt.subplot(gs[2:4,0:3])
##                    # ax4 = plt.subplot(gs[2:4,3:])
##                    # xlables = [round(i,2) for i in list(np.linspace(np.min(lon),np.max(lon),5))]
##                    # ylabels = [round(i,2) for i  in list(np.linspace(np.max(lat),np.min(lat),5))]
##                    # plt.setp([ax1,ax2,ax3],xticks=list(np.linspace(0,64,5)), xticklabels=xlables ,yticks=list(np.linspace(0,64,5)),yticklabels=ylabels)
##                    # ax1.title.set_text("(a) Ground Truth")
##                    # ax2.title.set_text("(b) SAVP")
##                    # ax3.title.set_text("(c) Diff.")
##                    # ax4.title.set_text("(d) MSE")
##                    #
##                    # ax1.xaxis.set_tick_params(labelsize=7)
##                    # ax1.yaxis.set_tick_params(labelsize = 7)
##                    # ax2.xaxis.set_tick_params(labelsize=7)
##                    # ax2.yaxis.set_tick_params(labelsize = 7)
##                    # ax3.xaxis.set_tick_params(labelsize=7)
##                    # ax3.yaxis.set_tick_params(labelsize = 7)
##                    #
##                    # init_images = np.zeros((input_images_.shape[1], input_images_.shape[2]))
##                    # print("inti images shape", init_images.shape)
##                    # xdata, ydata = [], []
##                    # #plot1 = ax1.imshow(init_images, cmap='jet', vmin =0, vmax = 1)
##                    # #plot2 = ax2.imshow(init_images, cmap='jet', vmin =0, vmax = 1)
##                    # plot1 = ax1.imshow(init_images, cmap='jet', vmin = 270, vmax = 300)
##                    # plot2 = ax2.imshow(init_images, cmap='jet', vmin = 270, vmax = 300)
##                    # #x = np.linspace(0, 64, 64)
##                    # #y = np.linspace(0, 64, 64)
##                    # #plot1 = ax1.contourf(x,y,init_images, cmap='jet', vmin = np.min(input_images), vmax = np.max(input_images))
##                    # #plot2 = ax2.contourf(x,y,init_images, cmap='jet', vmin = np.min(input_images), vmax = np.max(input_images))
##                    # fig.colorbar(plot1, ax=ax1).ax.tick_params(labelsize=7)
##                    # fig.colorbar(plot2, ax=ax2).ax.tick_params(labelsize=7)
##                    #
##                    # cm = LinearSegmentedColormap.from_list(
##                    #     cmap_name, "bwr", N = 5)
##                    #
##                    # plot3 = ax3.imshow(init_images, vmin=-20, vmax=20, cmap=cm)#cmap = 'PuBu_r',
##                    # #plot3 = ax3.imshow(init_images, vmin = -1, vmax = 1, cmap = cm)  # cmap = 'PuBu_r',
##                    # plot4, = ax4.plot([], [], color = "r")
##                    # ax4.set_xlim(0, future_length-1)
##                    # ax4.set_ylim(0, 20)
##                    # #ax4.set_ylim(0, 0.5)
##                    # ax4.set_xlabel("Frames", fontsize=10)
##                    # #ax4.set_ylabel("MSE", fontsize=10)
##                    # ax4.xaxis.set_tick_params(labelsize=7)
##                    # ax4.yaxis.set_tick_params(labelsize=7)
##                    #
##                    #
##                    # plots = [plot1, plot2, plot3, plot4]
##                    #
##                    # #fig.colorbar(plots[1], ax = [ax1, ax2])
##                    #
##                    # fig.colorbar(plots[2], ax=ax3).ax.tick_params(labelsize=7)
##                    # #fig.colorbar(plot1[0], ax=ax1).ax.tick_params(labelsize=7)
##                    # #fig.colorbar(plot2[1], ax=ax2).ax.tick_params(labelsize=7)
##                    #
##                    # def animation_sample(t):
##                    #     input_image = input_images_[t, :, :, 0]* (321.46630859375-235.2141571044922) + 235.2141571044922
##                    #     gen_image = gen_images_[t, :, :, 0]* (321.46630859375-235.2141571044922) + 235.2141571044922
##                    #     diff_image = input_gen_diff[t,:,:]
##                    #     # p = sns.lineplot(x=x,y=data,color="b")
##                    #     # p.tick_params(labelsize=17)
##                    #     # plt.setp(p.lines, linewidth=6)
##                    #     plots[0].set_data(input_image)
##                    #     plots[1].set_data(gen_image)
##                    #     #plots[0] = ax1.contourf(x, y, input_image, cmap = 'jet', vmin = np.min(input_images),vmax = np.max(input_images))
##                    #     #plots[1] = ax2.contourf(x, y, gen_image, cmap = 'jet', vmin = np.min(input_images),vmax = np.max(input_images))
##                    #     plots[2].set_data(diff_image)
##                    #
##                    #     if t >= future_length:
##                    #         #data = gen_mse_avg_[:t + 1]
##                    #         # x = list(range(len(gen_mse_avg_)))[:t+1]
##                    #         xdata.append(t-future_length)
##                    #         print("xdata", xdata)
##                    #         ydata.append(gen_mse_avg_[t])
##                    #         print("ydata", ydata)
##                    #         plots[3].set_data(xdata, ydata)
##                    #         fig.suptitle("Predicted Frame " + str(t-future_length))
##                    #     else:
##                    #         #plots[3].set_data(xdata, ydata)
##                    #         fig.suptitle("Context Frame " + str(t))
##                    #     return plots
##                    #
##                    # ani = animation.FuncAnimation(fig, animation_sample, frames=len(gen_mse_avg_), interval = 1000,
##                    #                               repeat_delay=2000)
##                    # ani.save(os.path.join(args.output_png_dir, "Sample_" + str(name) + ".mp4"))
##
####                else:
####                    pass
##





    #         # for i, gen_mse_avg_ in enumerate(gen_mse_avg):
    #         #     ims = []
    #         #     fig = plt.figure()
    #         #     plt.xlim(0,len(gen_mse_avg_))
    #         #     plt.ylim(np.min(gen_mse_avg),np.max(gen_mse_avg))
    #         #     plt.xlabel("Frames")
    #         #     plt.ylabel("MSE_AVG")
    #         #     #X = list(range(len(gen_mse_avg_)))
    #         #     #for t, gen_mse_avg_ in enumerate(gen_mse_avg):
    #         #     def animate_metric(j):
    #         #         data = gen_mse_avg_[:(j+1)]
    #         #         x = list(range(len(gen_mse_avg_)))[:(j+1)]
    #         #         p = sns.lineplot(x=x,y=data,color="b")
    #         #         p.tick_params(labelsize=17)
    #         #         plt.setp(p.lines, linewidth=6)
    #         #     ani = animation.FuncAnimation(fig, animate_metric, frames=len(gen_mse_avg_), interval = 1000, repeat_delay=2000)
    #         #     ani.save(os.path.join(args.output_png_dir, "MSE_AVG" + str(i) + ".gif"))
    #         #
    #         #
    #         # for i, input_images_ in enumerate(input_images):
    #         #     #context_images_ = (input_results['images'][i])
    #         #     #gen_images_fname = 'gen_image_%05d_%02d.gif' % (sample_ind + i, stochastic_sample_ind)
    #         #     ims = []
    #         #     fig = plt.figure()
    #         #     for t, input_image in enumerate(input_images_):
    #         #         im = plt.imshow(input_images[i, t, :, :, 0], interpolation = 'none')
    #         #         ttl = plt.text(1.5, 2,"Frame_" + str(t))
    #         #         ims.append([im,ttl])
    #         #     ani = animation.ArtistAnimation(fig, ims, interval= 1000, blit=True,repeat_delay=2000)
    #         #     ani.save(os.path.join(args.output_png_dir,"groud_true_images_" + str(i) + ".gif"))
    #         #     #plt.show()
    #         #
    #         # for i,gen_images_ in enumerate(gen_images):
    #         #     ims = []
    #         #     fig = plt.figure()
    #         #     for t, gen_image in enumerate(gen_images_):
    #         #         im = plt.imshow(gen_images[i, t, :, :, 0], interpolation = 'none')
    #         #         ttl = plt.text(1.5, 2, "Frame_" + str(t))
    #         #         ims.append([im, ttl])
    #         #     ani = animation.ArtistAnimation(fig, ims, interval = 1000, blit = True, repeat_delay = 2000)
    #         #     ani.save(os.path.join(args.output_png_dir, "prediction_images_" + str(i) + ".gif"))
    #
    #
    #             # for i, gen_images_ in enumerate(gen_images):
    #             #     #context_images_ = (input_results['images'][i] * 255.0).astype(np.uint8)
    #             #     #gen_images_ = (gen_images_ * 255.0).astype(np.uint8)
    #             #     #bing
    #             #     context_images_ = (input_results['images'][i])
    #             #     gen_images_fname = 'gen_image_%05d_%02d.gif' % (sample_ind + i, stochastic_sample_ind)
    #             #     context_and_gen_images = list(context_images_[:context_frames]) + list(gen_images_)
    #             #     plt.figure(figsize = (10,2))
    #             #     gs = gridspec.GridSpec(2,10)
    #             #     gs.update(wspace=0.,hspace=0.)
    #             #     for t, gen_image in enumerate(gen_images_):
    #             #         gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2,len(str(len(gen_images_) - 1)))
    #             #         gen_image_fname = gen_image_fname_pattern % (sample_ind + i, stochastic_sample_ind, t)
    #             #         plt.subplot(gs[t])
    #             #         plt.imshow(input_images[i, t, :, :, 0], interpolation = 'none')  # the last index sets the channel. 0 = t2
    #             #         # plt.pcolormesh(X_test[i,t,::-1,:,0], shading='bottom', cmap=plt.cm.jet)
    #             #         plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False,
    #             #                         right = False, labelbottom = False, labelleft = False)
    #             #         if t == 0: plt.ylabel('Actual', fontsize = 10)
    #             #
    #             #         plt.subplot(gs[t + 10])
    #             #         plt.imshow(gen_images[i, t, :, :, 0], interpolation = 'none')
    #             #         # plt.pcolormesh(X_hat[i,t,::-1,:,0], shading='bottom', cmap=plt.cm.jet)
    #             #         plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False,
    #             #                         right = False, labelbottom = False, labelleft = False)
    #             #         if t == 0: plt.ylabel('Predicted', fontsize = 10)
    #             #     plt.savefig(os.path.join(args.output_png_dir, gen_image_fname) + 'plot_' + str(i) + '.png')
    #             #     plt.clf()
    #
    #             # if args.gif_length:
    #             #     context_and_gen_images = context_and_gen_images[:args.gif_length]
    #             # save_gif(os.path.join(args.output_gif_dir, gen_images_fname),
    #             #          context_and_gen_images, fps=args.fps)
    #             #
    #             # gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2, len(str(len(gen_images_) - 1)))
    #             # for t, gen_image in enumerate(gen_images_):
    #             #     gen_image_fname = gen_image_fname_pattern % (sample_ind + i, stochastic_sample_ind, t)
    #             #     if gen_image.shape[-1] == 1:
    #             #       gen_image = np.tile(gen_image, (1, 1, 3))
    #             #     else:
    #             #       gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
    #             #     cv2.imwrite(os.path.join(args.output_png_dir, gen_image_fname), gen_image)
