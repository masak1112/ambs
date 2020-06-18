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
import seaborn as sns
import pickle
from random import seed
import random
import json
import numpy as np
#from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from video_prediction import datasets, models
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from video_prediction.utils.ffmpeg_gif import save_gif

with open("./splits_size_64_64_1/geo_info.json","r") as json_file:
    geo = json.load(json_file)
    lat = [round(i,2) for i in geo["lat"]]
    lon = [round(i,2) for i in geo["lon"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--results_dir", type=str, default='results', help="ignored if output_gif_dir is specified")
    parser.add_argument("--results_gif_dir", type=str, help="default is results_dir. ignored if output_gif_dir is specified")
    parser.add_argument("--results_png_dir", type=str, help="default is results_dir. ignored if output_png_dir is specified")
    parser.add_argument("--output_gif_dir", help="output directory where samples are saved as gifs. default is "
                                                 "results_gif_dir/model_fname")
    parser.add_argument("--output_png_dir", help="output directory where samples are saved as pngs. default is "
                                                 "results_png_dir/model_fname")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")

    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val', help='mode for dataset, val or test.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")

    parser.add_argument("--batch_size", type=int, default=8, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--num_stochastic_samples", type=int, default=1) #Bing original is 5, change to 1
    parser.add_argument("--gif_length", type=int, help="default is sequence_length")
    parser.add_argument("--fps", type=int, default=4)

    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    args.results_gif_dir = args.results_gif_dir or args.results_dir
    args.results_png_dir = args.results_png_dir or args.results_dir
    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
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
        args.output_gif_dir = args.output_gif_dir or os.path.join(args.results_gif_dir, os.path.split(checkpoint_dir)[1])
        args.output_png_dir = args.output_png_dir or os.path.join(args.results_png_dir, os.path.split(checkpoint_dir)[1])
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        args.output_gif_dir = args.output_gif_dir or os.path.join(args.results_gif_dir, 'model.%s' % args.model)
        args.output_png_dir = args.output_png_dir or os.path.join(args.results_png_dir, 'model.%s' % args.model)

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(
        args.input_dir,
        mode=args.mode,
        num_epochs=args.num_epochs,
        seed=args.seed,
        hparams_dict=dataset_hparams_dict,
        hparams=args.dataset_hparams)
    VideoPredictionModel = models.get_model_class(args.model)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': dataset.hparams.context_frames,
        'sequence_length': dataset.hparams.sequence_length,
        'repeat': dataset.hparams.time_shift,
    })
    model = VideoPredictionModel(
        mode=args.mode,
        hparams_dict=hparams_dict,
        hparams=args.model_hparams)

    sequence_length = model.hparams.sequence_length
    context_frames = model.hparams.context_frames
    future_length = sequence_length - context_frames

    if args.num_samples:
        if args.num_samples > dataset.num_examples_per_epoch():
            raise ValueError('num_samples cannot be larger than the dataset')
        num_examples_per_epoch = args.num_samples
    else:
        #Bing: error occurs here, cheats a little bit here
        #num_examples_per_epoch = dataset.num_examples_per_epoch()
        num_examples_per_epoch = args.batch_size * 8
    if num_examples_per_epoch % args.batch_size != 0:
        #bing
        #raise ValueError('batch_size should evenly divide the dataset size %d' % num_examples_per_epoch)
        pass
    #Bing if it is era 5 data we used dataset.make_batch_v2
    #inputs = dataset.make_batch(args.batch_size)
    inputs = dataset.make_batch(args.batch_size)
    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
    with tf.variable_scope(''):
        model.build_graph(input_phs)

    for output_dir in (args.output_gif_dir, args.output_png_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(args), sort_keys=True, indent=4))
        with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
            f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(output_dir, "model_hparams.json"), "w") as f:
            f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()
    model.restore(sess, args.checkpoint)
    sample_ind = 0
    gen_images_all = []
    input_images_all = []

    while True:
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        print("evaluation samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))
        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        for stochastic_sample_ind in range(args.num_stochastic_samples): #Todo: why use here
            print("Stochastic sample id", stochastic_sample_ind)
            gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)
            #input_images = sess.run(inputs["images"])
            #Bing: Add evaluation metrics
            # fetches = {'images': model.inputs['images']}
            # fetches.update(model.eval_outputs.items())
            # fetches.update(model.eval_metrics.items())
            # results = sess.run(fetches, feed_dict = feed_dict)
            # input_images = results["images"] #shape (batch_size,future_frames,height,width,channel)
            # only keep the future frames
            #gen_images = gen_images[:, -future_length:] #(8,10,64,64,1) (batch_size, sequences, height, width, channel)
            #input_images = input_results["images"][:,-future_length:,:,:]
            input_images = input_results["images"][:,1:,:,:,:]
            #gen_mse_avg = results["eval_mse/avg"] #shape (batch_size,future_frames)
            print("Finish sample ind",stochastic_sample_ind)
            input_gen_diff_ = input_images - gen_images
            #diff_image_range = pd.cut(input_gen_diff_.flatten(), bins = 4, labels = [-10, -5, 0, 5], right = False)
            #diff_image_range = np.reshape(np.array(diff_image_range),input_gen_diff_.shape)
            gen_images_all.extend(gen_images)
            input_images_all.extend(input_images)

            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            cmap_name = 'my_list'
            if sample_ind < 100:
                for i in range(len(gen_images)):
                    name = 'Batch_id_' + str(sample_ind) + " + Sample_" + str(i)
                    gen_images_ = gen_images[i, :]
                    gen_mse_avg_ = [np.mean(input_gen_diff_[i, frame, :, :, :]**2) for frame in
                                    range(19)]  # return the list with 10 (sequence) mse
                    input_gen_diff = input_gen_diff_ [i,:,:,:,:]
                    input_images_ = input_images[i, :]
                    #gen_mse_avg_ = gen_mse_avg[i, :]

                    # Bing: This is to check the difference between the images and next images for debugging the freezon issues
                    # gen_images_diff = []
                    # for gen_idx in range(len(gen_images_) - 1):
                    #     img_1 = gen_images_[gen_idx, :, :, :]
                    #     img_2 = gen_images_[gen_idx + 1, :, :, :]
                    #     img_diff = img_2 - img_1
                    #     img_diff_nonzero = [e for img_idx, e in enumerate(img_diff.flatten()) if round(e,3) != 0.000]
                    #     gen_images_diff.append(img_diff_nonzero)

                    fig = plt.figure()
                    gs = gridspec.GridSpec(4,6)
                    gs.update(wspace = 0.7,hspace=0.8)
                    ax1 = plt.subplot(gs[0:2,0:3])
                    ax2 = plt.subplot(gs[0:2,3:],sharey=ax1)
                    ax3 = plt.subplot(gs[2:4,0:3])
                    ax4 = plt.subplot(gs[2:4,3:])
                    xlables = [round(i,2) for i in list(np.linspace(np.min(lon),np.max(lon),5))]
                    ylabels = [round(i,2) for i  in list(np.linspace(np.max(lat),np.min(lat),5))]
                    plt.setp([ax1,ax2,ax3],xticks=list(np.linspace(0,64,5)), xticklabels=xlables ,yticks=list(np.linspace(0,64,5)),yticklabels=ylabels)
                    ax1.title.set_text("(a) Ground Truth")
                    ax2.title.set_text("(b) SAVP")
                    ax3.title.set_text("(c) Diff.")
                    ax4.title.set_text("(d) MSE")

                    ax1.xaxis.set_tick_params(labelsize=7)
                    ax1.yaxis.set_tick_params(labelsize = 7)
                    ax2.xaxis.set_tick_params(labelsize=7)
                    ax2.yaxis.set_tick_params(labelsize = 7)
                    ax3.xaxis.set_tick_params(labelsize=7)
                    ax3.yaxis.set_tick_params(labelsize = 7)

                    init_images = np.zeros((input_images_.shape[1], input_images_.shape[2]))
                    print("inti images shape", init_images.shape)
                    xdata, ydata = [], []
                    plot1 = ax1.imshow(init_images, cmap='jet', vmin = 270, vmax = 300)
                    plot2 = ax2.imshow(init_images, cmap='jet', vmin = 270, vmax = 300)
                    #x = np.linspace(0, 64, 64)
                    #y = np.linspace(0, 64, 64)
                    #plot1 = ax1.contourf(x,y,init_images, cmap='jet', vmin = np.min(input_images), vmax = np.max(input_images))
                    #plot2 = ax2.contourf(x,y,init_images, cmap='jet', vmin = np.min(input_images), vmax = np.max(input_images))
                    fig.colorbar(plot1, ax=ax1).ax.tick_params(labelsize=7)
                    fig.colorbar(plot2, ax=ax2).ax.tick_params(labelsize=7)

                    cm = LinearSegmentedColormap.from_list(
                        cmap_name, "bwr", N = 5)

                    plot3 = ax3.imshow(init_images, vmin=-10, vmax=10, cmap=cm)#cmap = 'PuBu_r',
                    plot4, = ax4.plot([], [], color = "r")
                    ax4.set_xlim(0, len(gen_mse_avg_)-1)
                    ax4.set_ylim(0, 10)
                    ax4.set_xlabel("Frames", fontsize=10)
                    #ax4.set_ylabel("MSE", fontsize=10)
                    ax4.xaxis.set_tick_params(labelsize=7)
                    ax4.yaxis.set_tick_params(labelsize=7)


                    plots = [plot1, plot2, plot3, plot4]

                    #fig.colorbar(plots[1], ax = [ax1, ax2])

                    fig.colorbar(plots[2], ax=ax3).ax.tick_params(labelsize=7)
                    #fig.colorbar(plot1[0], ax=ax1).ax.tick_params(labelsize=7)
                    #fig.colorbar(plot2[1], ax=ax2).ax.tick_params(labelsize=7)

                    def animation_sample(t):
                        input_image = input_images_[t, :, :, 0]
                        gen_image = gen_images_[t, :, :, 0]
                        diff_image = input_gen_diff[t,:,:,0]

                        data = gen_mse_avg_[:t + 1]
                        # x = list(range(len(gen_mse_avg_)))[:t+1]
                        xdata.append(t)
                        print("xdata", xdata)
                        ydata.append(gen_mse_avg_[t])

                        print("ydata", ydata)
                        # p = sns.lineplot(x=x,y=data,color="b")
                        # p.tick_params(labelsize=17)
                        # plt.setp(p.lines, linewidth=6)
                        plots[0].set_data(input_image)
                        plots[1].set_data(gen_image)
                        #plots[0] = ax1.contourf(x, y, input_image, cmap = 'jet', vmin = np.min(input_images),vmax = np.max(input_images))
                        #plots[1] = ax2.contourf(x, y, gen_image, cmap = 'jet', vmin = np.min(input_images),vmax = np.max(input_images))
                        plots[2].set_data(diff_image)
                        plots[3].set_data(xdata, ydata)
                        fig.suptitle("Frame " + str(t+1))

                        return plots

                    ani = animation.FuncAnimation(fig, animation_sample, frames=len(gen_mse_avg_), interval = 1000,
                                                  repeat_delay = 2000)
                    ani.save(os.path.join(args.output_png_dir, "Sample_" + str(name) + ".mp4"))

            else:
                pass

















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

        sample_ind += args.batch_size


    with open(os.path.join(args.output_png_dir, "input_images_all"),"wb") as input_files:
        pickle.dump(input_images_all,input_files)

    with open(os.path.join(args.output_png_dir, "gen_images_all"),"wb") as gen_files:
        pickle.dump(gen_images_all,gen_files)

    with open(os.path.join(args.output_png_dir, "input_images_all"),"rb") as input_files:
        input_images_all = pickle.load(input_files)

    with open(os.path.join(args.output_png_dir, "gen_images_all"),"rb") as gen_files:
        gen_images_all=pickle.load(gen_files)
    ims = []
    fig = plt.figure()
    for frame in range(19):
        input_gen_diff = np.mean((np.array(gen_images_all) - np.array(input_images_all))**2, axis = 0)[frame, :,:, 0] # Get the first prediction frame (batch,height, width, channel)
        #pix_mean = np.mean(input_gen_diff, axis = 0)
        #pix_std = np.std(input_gen_diff, axis=0)
        im = plt.imshow(input_gen_diff, interpolation = 'none',cmap='PuBu')
        if frame == 0:
            fig.colorbar(im)
        ttl = plt.text(1.5, 2, "Frame_" + str(frame +1))
        ims.append([im, ttl])
    ani = animation.ArtistAnimation(fig, ims, interval = 1000, blit = True, repeat_delay = 2000)
    ani.save(os.path.join(args.output_png_dir, "Mean_Frames.mp4"))
    plt.close("all")

    ims = []
    fig = plt.figure()
    for frame in range(19):
        pix_std= np.std((np.array(gen_images_all) - np.array(input_images_all))**2, axis = 0)[frame, :,:, 0]  # Get the first prediction frame (batch,height, width, channel)
        #pix_mean = np.mean(input_gen_diff, axis = 0)
        #pix_std = np.std(input_gen_diff, axis=0)
        im = plt.imshow(pix_std, interpolation = 'none',cmap='PuBu')
        if frame == 0:
            fig.colorbar(im)
        ttl = plt.text(1.5, 2, "Frame_" + str(frame+1))
        ims.append([im, ttl])
    ani = animation.ArtistAnimation(fig, ims, interval = 1000, blit = True, repeat_delay = 2000)
    ani.save(os.path.join(args.output_png_dir, "Std_Frames.mp4"))

    gen_images_all = np.array(gen_images_all)
    input_images_all = np.array(input_images_all)
    # mse_model = np.mean((input_images_all[:, 1:,:,:,0] - gen_images_all[:, 1:,:,:,0])**2)  # look at all timesteps except the first
    # mse_model_last = np.mean((input_images_all[:, future_length-1,:,:,0] - gen_images_all[:, future_length-1,:,:,0])**2)
    # mse_prev = np.mean((input_images_all[:, :-1,:,:,0] - gen_images_all[:, 1:,:,:,0])**2 )

    mse_model = np.mean((input_images_all[:, :10,:,:,0] - gen_images_all[:, :10,:,:,0])**2)  # look at all timesteps except the first
    mse_model_last = np.mean((input_images_all[:,10,:,:,0] - gen_images_all[:, 10,:,:,0])**2)
    mse_prev = np.mean((input_images_all[:, :9, :, :, 0] - input_images_all[:, 1:10, :, :, 0])**2 )

    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    psnr_model = psnr(input_images_all[:, :10, :, :, 0],  gen_images_all[:, :10, :, :, 0])
    psnr_model_last = psnr(input_images_all[:, 10, :, :, 0],  gen_images_all[:,10, :, :, 0])
    psnr_prev = psnr(input_images_all[:, :9, :, :, 0],  input_images_all[:, 1:10, :, :, 0])
    f = open(os.path.join(args.output_png_dir,'prediction_scores_4prediction.txt'), 'w')
    f.write("Model MSE: %f\n" % mse_model)
    f.write("Model MSE from only last prediction in sequence: %f\n" % mse_model_last)
    f.write("Previous Frame MSE: %f\n" % mse_prev)
    f.write("Model PSNR: %f\n" % psnr_model)
    f.write("Model PSNR from only last prediction in sequence: %f\n" % psnr_model_last)
    f.write("Previous frame PSNR: %f\n" % psnr_prev)
    f.write("Shape of X_test: " + str(input_images_all.shape))
    f.write("")
    f.write("Shape of X_hat: " + str(gen_images_all.shape))
    f.close()

    seed(1)
    s = random.sample(range(len(gen_images_all)), 100)
    print("******KDP******")
    #kernel density plot for checking the model collapse
    fig = plt.figure()
    kdp = sns.kdeplot(gen_images_all[s].flatten(), shade=True, color="r", label = "Generate Images")
    kdp = sns.kdeplot(input_images_all[s].flatten(), shade=True, color="b", label = "Ground True")
    kdp.set(xlabel = 'Temperature (K)', ylabel = 'Probability')
    plt.savefig(os.path.join(args.output_png_dir, "kdp_gen_images.png"), dpi = 400)
    plt.clf()

    #line plot for evaluating the prediction and groud-truth
    for i in [0,3,6,9,12,15,18]:
        fig = plt.figure()
        plt.scatter(gen_images_all[:,i,:,:][s].flatten(),input_images_all[:,i,:,:][s].flatten(),s=0.3)
        #plt.scatter(gen_images_all[:,0,:,:].flatten(),input_images_all[:,0,:,:].flatten(),s=0.3)
        plt.xlabel("Prediction")
        plt.ylabel("Real values")
        plt.title("Frame_{}".format(i+1))
        plt.plot([250,300], [250,300],color="black")
        plt.savefig(os.path.join(args.output_png_dir,"pred_real_frame_{}.png".format(str(i))))
        plt.clf()


    mse_model_by_frames = np.mean((input_images_all[:, :, :, :, 0][s] - gen_images_all[:, :, :, :, 0][s]) ** 2,axis=(2,3)) #return (batch, sequence)
    x = [str(i+1) for i in list(range(19))]
    fig,axis = plt.subplots()
    mean_f = np.mean(mse_model_by_frames, axis = 0)
    median = np.median(mse_model_by_frames, axis=0)
    q_low = np.quantile(mse_model_by_frames, q=0.25, axis=0)
    q_high = np.quantile(mse_model_by_frames, q=0.75, axis=0)
    d_low = np.quantile(mse_model_by_frames,q=0.1, axis=0)
    d_high = np.quantile(mse_model_by_frames, q=0.9, axis=0)
    plt.fill_between(x, d_high, d_low, color="ghostwhite",label="interdecile range")
    plt.fill_between(x,q_high, q_low , color = "lightgray", label="interquartile range")
    plt.plot(x, median, color="grey", linewidth=0.6, label="Median")
    plt.plot(x, mean_f, color="peachpuff",linewidth=1.5, label="Mean")
    plt.title(f'MSE percentile')
    plt.xlabel("Frames")
    plt.legend(loc=2, fontsize=8)
    plt.savefig(os.path.join(args.output_png_dir,"mse_percentiles.png"))

if __name__ == '__main__':
    main()
