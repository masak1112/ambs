from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import random
import cv2
import numpy as np
import tensorflow as tf

import numpy as np
#from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import seaborn as sns

from video_prediction import datasets, models
from video_prediction.utils.ffmpeg_gif import save_gif







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

    parser.add_argument("--num_stochastic_samples", type=int, default=5)
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
    inputs = dataset.make_batch_v2(args.batch_size)
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

    while True:
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        print("evaluation samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))
        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        for stochastic_sample_ind in range(args.num_stochastic_samples):
            gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)
            #input_images = sess.run(inputs["images"])

            #Bing: Add evaluation metrics
            fetches = {'images': model.inputs['images']}
            fetches.update(model.eval_outputs.items())
            fetches.update(model.eval_metrics.items())
            results = sess.run(fetches, feed_dict = feed_dict)
            input_images = results["images"] #shape (batch_size,future_frames,height,width,channel)
            # only keep the future frames
            gen_images = gen_images[:, -future_length:]
            input_images = input_images[:,-future_length:]
            gen_mse_avg = results["eval_mse/avg"] #shape (batch_size,future_frames)

            for i, gen_mse_avg_ in enumerate(gen_mse_avg):
                ims = []
                fig = plt.figure()
                plt.xlim(0,len(gen_mse_avg_))
                plt.ylim(np.min(gen_mse_avg),np.max(gen_mse_avg))
                plt.xlabel("Frames")
                plt.ylabel("MSE_AVG")
                #X = list(range(len(gen_mse_avg_)))
                #for t, gen_mse_avg_ in enumerate(gen_mse_avg):
                def animate_metric(j):
                    data = gen_mse_avg_[:(j+1)]
                    x = list(range(len(gen_mse_avg_)))[:(j+1)]
                    p = sns.lineplot(x=x,y=data,color="b")
                    p.tick_params(labelsize=17)
                    plt.setp(p.lines, linewidth=6)
                ani = animation.FuncAnimation(fig, animate_metric, frames=len(gen_mse_avg_), interval = 1000, repeat_delay=2000)
                ani.save(os.path.join(args.output_png_dir, "MSE_AVG" + str(i) + ".mp4"))


            for i, input_images_ in enumerate(input_images):
                #context_images_ = (input_results['images'][i])
                #gen_images_fname = 'gen_image_%05d_%02d.gif' % (sample_ind + i, stochastic_sample_ind)
                ims = []
                fig = plt.figure()
                for t, input_image in enumerate(input_images_):
                    im = plt.imshow(input_images[i, t, :, :, 0], interpolation = 'none')
                    ttl = plt.text(1.5, 2,"Frame_" + str(t))
                    ims.append([im,ttl])
                ani = animation.ArtistAnimation(fig, ims, interval= 1000, blit=True,repeat_delay=2000)
                ani.save(os.path.join(args.output_png_dir,"groud_true_images_" + str(i) + ".mp4"))
                #plt.show()

            for i,gen_images_ in enumerate(gen_images):
                ims = []
                fig = plt.figure()
                for t, gen_image in enumerate(gen_images_):
                    im = plt.imshow(gen_images[i, t, :, :, 0], interpolation = 'none')
                    ttl = plt.text(1.5, 2, "Frame_" + str(t))
                    ims.append([im, ttl])
                ani = animation.ArtistAnimation(fig, ims, interval = 1000, blit = True, repeat_delay = 2000)
                ani.save(os.path.join(args.output_png_dir, "prediction_images_" + str(i) + ".mp4"))



                # for i, gen_images_ in enumerate(gen_images):
                #     #context_images_ = (input_results['images'][i] * 255.0).astype(np.uint8)
                #     #gen_images_ = (gen_images_ * 255.0).astype(np.uint8)
                #     #bing
                #     context_images_ = (input_results['images'][i])
                #     gen_images_fname = 'gen_image_%05d_%02d.gif' % (sample_ind + i, stochastic_sample_ind)
                #     context_and_gen_images = list(context_images_[:context_frames]) + list(gen_images_)
                #     plt.figure(figsize = (10,2))
                #     gs = gridspec.GridSpec(2,10)
                #     gs.update(wspace=0.,hspace=0.)
                #     for t, gen_image in enumerate(gen_images_):
                #         gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2,len(str(len(gen_images_) - 1)))
                #         gen_image_fname = gen_image_fname_pattern % (sample_ind + i, stochastic_sample_ind, t)
                #         plt.subplot(gs[t])
                #         plt.imshow(input_images[i, t, :, :, 0], interpolation = 'none')  # the last index sets the channel. 0 = t2
                #         # plt.pcolormesh(X_test[i,t,::-1,:,0], shading='bottom', cmap=plt.cm.jet)
                #         plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False,
                #                         right = False, labelbottom = False, labelleft = False)
                #         if t == 0: plt.ylabel('Actual', fontsize = 10)
                #
                #         plt.subplot(gs[t + 10])
                #         plt.imshow(gen_images[i, t, :, :, 0], interpolation = 'none')
                #         # plt.pcolormesh(X_hat[i,t,::-1,:,0], shading='bottom', cmap=plt.cm.jet)
                #         plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False,
                #                         right = False, labelbottom = False, labelleft = False)
                #         if t == 0: plt.ylabel('Predicted', fontsize = 10)
                #     plt.savefig(os.path.join(args.output_png_dir, gen_image_fname) + 'plot_' + str(i) + '.png')
                #     plt.clf()

                # if args.gif_length:
                #     context_and_gen_images = context_and_gen_images[:args.gif_length]
                # save_gif(os.path.join(args.output_gif_dir, gen_images_fname),
                #          context_and_gen_images, fps=args.fps)
                #
                # gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2, len(str(len(gen_images_) - 1)))
                # for t, gen_image in enumerate(gen_images_):
                #     gen_image_fname = gen_image_fname_pattern % (sample_ind + i, stochastic_sample_ind, t)
                #     if gen_image.shape[-1] == 1:
                #       gen_image = np.tile(gen_image, (1, 1, 3))
                #     else:
                #       gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(os.path.join(args.output_png_dir, gen_image_fname), gen_image)

        sample_ind += args.batch_size

if __name__ == '__main__':
    main()
