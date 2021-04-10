from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Yan Ji, Michael Langguth"
__date__ = "2020-11-10"

import argparse
import os
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
import warnings
import pickle
from random import seed
import datetime
import json
from netCDF4 import Dataset, date2num
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
from normalization import Norm_data
from metadata import MetaData as MetaData
from main_scripts.main_train_models import *
from data_preprocess.preprocess_data_step2 import *
import shutil
from model_modules.video_prediction import datasets, models, metrics


class Postprocess(TrainModel):
    def __init__(self, results_dir=None, checkpoint=None, mode="test",
                 batch_size=None, num_samples=None, num_stochastic_samples=1, stochastic_plot_id=0,
                 gpu_mem_frac=None, seed=None, args=None, run_mode="deterministic"):
        """
        The function for inference, generate results and images
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
        run_mode           :str, if "deterministic" then the model running for deterministic forecasting,  other string values, it will go for stochastic forecasting

        Side notes : other important varialbes in the class:
        self.ts               : list, contains the sequence_length timestamps
        self.gen_images_      :  the length of generate images by model is sequence_length - 1
        self.persistent_image : the length of persistent images is sequence_length - 1
        self.input_images     : the length of inputs images is sequence length

        """

        # initialize input directories (to be retrieved by load_jsons)
        self.input_dir = None
        self.input_dir_tfr = None
        self.input_dir_pkl = None
        # initialize simple evalualtion metrics for model and persistence forecasts
        # (calculated when executing run-method)
        self.prst_metric_mse_all, self.prst_metric_psnr_all = None, None
        self.fcst_metric_mse_all, self.fcst_metric_psnr_all = None, None
        # initialze list tracking initialization time of generated forecasts
        self.ts_fcst_ini = []
        # set further attributes from parsed arguments
        self.results_dir = self.output_dir = os.path.normpath(results_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.batch_size = batch_size
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.num_samples = num_samples
        self.num_stochastic_samples = num_stochastic_samples
        self.stochastic_plot_id = stochastic_plot_id
        self.args = args
        self.checkpoint = checkpoint
        self.run_mode = run_mode
        self.mode = mode
        if self.num_samples < self.batch_size:
            raise ValueError("The number of samples should be at least as large as the batch size. " +
                             "Currently, number of samples: {} batch size: {}"
                             .format(self.num_samples, self.batch_size))
        if self.checkpoint is None:
            raise ValueError("The directory point to checkpoint is empty, must be provided for postprocess step")

        if not os.path.isdir(self.checkpoint):
            raise NotADirectoryError("The checkpoint-directory '{0}' does not exist".format(self.checkpoint))

    def __call__(self):
        self.set_seed()
        self.save_args_to_option_json()
        self.copy_data_model_json()
        self.load_jsons()
        self.get_metadata()
        self.setup_test_dataset()
        self.setup_model()
        self.get_data_params()
        self.setup_num_samples_per_epoch()
        self.get_stat_file()
        self.make_test_dataset_iterator()
        self.check_stochastic_samples_ind_based_on_model()
        self.setup_graph()
        self.setup_gpu_config()

    def save_args_to_option_json(self):
        """
        Save the argments defined by user to the results dir
        """
        with open(os.path.join(self.results_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(self.args), sort_keys=True, indent=4))

    def copy_data_model_json(self):
        """
        Copy relevant JSON-files from checkpoints directory to results_dir
        """
        method_name = Postprocess.copy_data_model_json.__name__

        # correctness of self.checkpoint and self.results_dir is already checked in __init__
        model_opt_js = os.path.join(self.checkpoint, "options.json")
        model_ds_js = os.path.join(self.checkpoint, "dataset_hparams.json")
        model_hp_js = os.path.join(self.checkpoint, "model_hparams.json")
        model_dd_js = os.path.join(self.checkpoint, "data_dict.json")

        if os.path.isfile(model_opt_js):
            shutil.copy(model_opt_js, os.path.join(self.results_dir, "options_checkpoints.json"))
        else:
            raise FileNotFoundError("%{0}: The file {1} does not exist".format(method_name, model_opt_js))

        if os.path.isfile(model_ds_js):
            shutil.copy(model_ds_js, os.path.join(self.results_dir, "dataset_hparams.json"))
        else:
            raise FileNotFoundError("%{0}: the file {1} does not exist".format(method_name, model_ds_js))

        if os.path.isfile(model_hp_js):
            shutil.copy(model_hp_js, os.path.join(self.results_dir, "model_hparams.json"))
        else:
            raise FileNotFoundError("%{0}: The file {1} does not exist".format(method_name, model_hp_js))

        if os.path.isfile(model_dd_js):
            shutil.copy(model_dd_js, os.path.join(self.results_dir, "data_dict.json"))
        else:
            raise FileNotFoundError("%{0}: The file {1} does not exist".format(method_name, model_dd_js))

    def load_jsons(self):
        """
        Set attributes pointing to JSON-files which track essential information and also load some information
        to store it to attributes of the class instance
        """
        method_name = Postprocess.load_jsons.__name__

        self.datasplit_dict = os.path.join(self.results_dir, "data_dict.json")
        self.model_hparams_dict = os.path.join(self.results_dir, "model_hparams.json")
        checkpoint_opt_dict = os.path.join(self.results_dir, "options_checkpoints.json")

        # sanity checks on the JSON-files
        if not os.path.isfile(self.datasplit_dict):
            raise FileNotFoundError("%{0}: The file data_dict.json is missing in {1}".format(method_name,
                                                                                             self.results_dir))

        if not os.path.isfile(self.model_hparams_dict):
            raise FileNotFoundError("%{0}: The file model_hparams.json is missing in {1}".format(method_name,
                                                                                                 self.results_dir))

        if not os.path.isfile(checkpoint_opt_dict):
            raise FileNotFoundError("%{0}: The file options_checkpoints.json is missing in {1}"
                                    .format(method_name, self.results_dir))
        # retrieve some data from options_checkpoints.json
        try:
            with open(checkpoint_opt_dict) as f:
                self.options_checkpoint = json.loads(f.read())
                self.dataset = self.options_checkpoint["dataset"]
                self.model = self.options_checkpoint["model"]
                self.input_dir_tfr = self.options_checkpoint["input_dir"]
                self.input_dir = os.path.dirname(self.input_dir_tfr)
                self.input_dir_pkl = os.path.join(self.input_dir, "pickle")
        except:
            raise IOError("%{0}: Could not retrieve all information from options_checkpoints.json".format(method_name))

        self.model_hparams_dict_load = self.get_model_hparams_dict()

    def get_metadata(self):

        method_name = Postprocess.get_metadata.__name__

        # some sanity checks
        if self.input_dir is None:
            raise AttributeError("%{0}: input_dir-attribute is still None".format(method_name))

        metadata_fl = os.path.join(self.input_dir, "metadata.json")

        if not os.path.isfile(metadata_fl):
            raise FileNotFoundError("%{0}: Could not find metadata JSON-file under '{1}'".format(method_name,
                                                                                                 self.input_dir))

        md_instance = MetaData(json_file=metadata_fl)

        try:
            self.height, self.width = md_instance.ny, md_instance.nx
            self.vars_in = md_instance.variables

            self.lats = xr.DataArray(md_instance.lat, coords={"lat": md_instance.lat}, dims="lat",
                                     attrs={"units": "degrees_east"})
            self.lons = xr.DataArray(md_instance.lon, coords={"lon": md_instance.lon}, dims="lon",
                                     attrs={"units": "degrees_north"})
        except:
            raise IOError("%{0}: Could not retrieve all required information from metadata-file '{1}'"
                          .format(method_name, metadata_fl))

    def setup_test_dataset(self):
        """
        setup the test dataset instance
        """
        VideoDataset = datasets.get_dataset_class(self.dataset)
        self.test_dataset = VideoDataset(input_dir=self.input_dir_tfr, mode=self.mode,
                                         datasplit_config=self.datasplit_dict)

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

    def get_stat_file(self):
        """
        Load the statistics from statistic file from the input directory
        """
        self.stat_fl = os.path.join(self.input_dir, "statistics.json")

    def make_test_dataset_iterator(self):
        """
        Make the dataset iterator
        """
        self.test_tf_dataset = self.test_dataset.make_dataset(self.batch_size)
        self.test_iterator = self.test_tf_dataset.make_one_shot_iterator()
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.test_handle = self.test_iterator.string_handle()
        self.iterator = tf.data.Iterator.from_string_handle(self.test_handle, self.test_tf_dataset.output_types,
                                                            self.test_tf_dataset.output_shapes)
        self.inputs = self.iterator.get_next()
        self.input_ts = self.inputs["T_start"]
        # if self.dataset == "era5" and self.model == "savp":
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
        self.sess = tf.Session(config=self.config)
        self.sess.graph.as_default()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def get_input_data_per_batch(self):
        """
        Get the input sequence from the dataset iterator object stored in self.inputs and denormalize the data
        :return input_results: the normalized input data
        :return input_images_denorm: the denormalized input data
        :return t_starts: the initial time of the sequences
        """
        method = Postprocess.get_input_data_per_batch.__name__

        input_results = self.sess.run(self.inputs)
        input_images = input_results["images"]
        t_starts = input_results["T_start"]
        # get one seq and the corresponding start time poin
        # self.t_starts = input_results["T_start"]
        input_images_denorm_all = []
        for batch_id in np.arange(self.batch_size):
            input_images_ = Postprocess.get_one_seq_from_batch(input_images, batch_id)
            # Denormalize input data
            #ts = Postprocess.generate_seq_timestamps(self.t_starts[batch_id], len_seq=self.sequence_length)
            input_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl, input_images_, self.vars_in)
            assert np.ndim(input_images_denorm) == 4, "%{0}: Data of input sequence must have four dimensions"\
                                                      .format(method)
            # ML: Do not plot in loop
            #Postprocess.plot_seq_imgs(imgs=input_images_denorm[self.context_frames:, :, :, 0],
            #                          lats=self.lats, lons=self.lons, ts=ts[self.context_frames:], label="Ground Truth",
            #                          output_png_dir=self.results_dir)
            input_images_denorm_all.append(list(input_images_denorm))
        assert np.ndim(np.array(input_images_denorm_all)) == 5, "%{0}: Data of all input ".format(method) + \
                                                                "sequences per mini-batch must be 5."
        return input_results, input_images_denorm_all, t_starts

    def run_stochastic(self):
        """
        Run session, save results to netcdf, plot input images, generate images and persistent images
        """
        method = Postprocess.run_stochastic.__name__
        raise ValueError("ML: %{0} is not runnable now".format(method))

        self.init_session()
        self.restore(self.sess, self.checkpoint)
        # Loop for samples
        self.sample_ind = 0
        self.prst_metric_all = []  # store evaluation metrics of persistence forecast (shape [future_len])
        self.fcst_metric_all = []  # store evaluation metric of stochastic forecasts (shape [nstoch, batch, future_len])
        while self.sample_ind < self.num_samples_per_epoch:
            if self.num_samples_per_epoch < self.sample_ind:
                break
            else:
                # run the inputs and plot each sequence images
                self.input_results, self.input_images_denorm_all, self.t_starts = self.get_input_data_per_batch()

            feed_dict = {input_ph: self.input_results[name] for name, input_ph in self.inputs.items()}
            gen_loss_stochastic_batch = []  # [stochastic_ind,future_length]
            gen_images_stochastic = []  # [stochastic_ind,batch_size,seq_len,lat,lon,channels]
            # Loop for stochastics
            for stochastic_sample_ind in range(self.num_stochastic_samples):
                print("stochastic_sample_ind:", stochastic_sample_ind)
                # return [batchsize,seq_len,lat,lon,channel]
                gen_images = self.sess.run(self.video_model.outputs['gen_images'], feed_dict=feed_dict)
                # The generate images seq_len should be sequence_len -1, since the last one is
                # not used for comparing with groud truth
                assert gen_images.shape[1] == self.sequence_length - 1
                gen_images_per_batch = []
                if stochastic_sample_ind == 0:
                    persistent_images_per_batch = []  # [batch_size,seq_len,lat,lon,channel]
                    ts_batch = []
                for i in range(self.batch_size):
                    # generate time stamps for sequences only once, since they are the same for all ensemble members
                    if stochastic_sample_ind == 0:
                        self.ts = Postprocess.generate_seq_timestamps(self.t_starts[i], len_seq=self.sequence_length)
                        init_date_str = self.ts[0].strftime("%Y%m%d%H")
                        ts_batch.append(init_date_str)
                        # get persistence_images
                        self.persistence_images, self.ts_persistence = Postprocess.get_persistence(self.ts,
                                                                                                   self.input_dir_pkl)
                        persistent_images_per_batch.append(self.persistence_images)
                        assert len(np.array(persistent_images_per_batch).shape) == 5
                        self.plot_persistence_images()

                    # Denormalized data for generate
                    gen_images_ = gen_images[i]
                    self.gen_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl, gen_images_,
                                                                                    self.vars_in)
                    gen_images_per_batch.append(self.gen_images_denorm)
                    assert len(np.array(gen_images_per_batch).shape) == 5
                    # only plot when the first stochastic ind otherwise too many plots would be created
                    # only plot the stochastic results of user-defined ind
                    self.plot_generate_images(stochastic_sample_ind, self.stochastic_plot_id)
                # calculate the persistnet error per batch
                if stochastic_sample_ind == 0:
                    persistent_loss_per_batch = Postprocess.calculate_metrics_by_batch(self.input_images_denorm_all,
                                                                                       persistent_images_per_batch,
                                                                                       self.future_length,
                                                                                       self.context_frames,
                                                                                       matric="mse", channel=0)
                    self.prst_metric_all.append(persistent_loss_per_batch)

                # calculate the gen_images_per_batch error
                gen_loss_per_batch = Postprocess.calculate_metrics_by_batch(self.input_images_denorm_all,
                                                                            gen_images_per_batch, self.future_length,
                                                                            self.context_frames,
                                                                            matric="mse", channel=0)
                gen_loss_stochastic_batch.append(
                    gen_loss_per_batch)  # self.gen_images_stochastic[stochastic,future_length]
                print("gen_images_per_batch shape:", np.array(gen_images_per_batch).shape)
                gen_images_stochastic.append(
                    gen_images_per_batch)  # [stochastic,batch_size, seq_len, lat, lon, channel]

                # Switch the 0 and 1 position
                print("before transpose:", np.array(gen_images_stochastic).shape)
            gen_images_stochastic = np.transpose(np.array(gen_images_stochastic), (
                1, 0, 2, 3, 4, 5))  # [batch_size, stochastic, seq_len, lat, lon, chanel]
            Postprocess.check_gen_images_stochastic_shape(gen_images_stochastic)
            assert len(gen_images_stochastic.shape) == 6
            assert np.array(gen_images_stochastic).shape[1] == self.num_stochastic_samples

            self.fcst_metric_all.append(
                gen_loss_stochastic_batch)  # [samples/batch_size,stochastic,future_length]
            # save input and stochastic generate images to netcdf file
            # For each prediction (either deterministic or ensemble) we create one netCDF file.
            for batch_id in range(self.batch_size):
                self.save_to_netcdf_for_stochastic_generate_images(self.input_images_denorm_all[batch_id],
                                                                   persistent_images_per_batch[batch_id],
                                                                   np.array(gen_images_stochastic)[batch_id],
                                                                   fl_name="vfp_date_{}_sample_ind_{}.nc" \
                                                                   .format(ts_batch[batch_id],
                                                                           self.sample_ind + batch_id))

            self.sample_ind += self.batch_size

        self.persistent_loss_all_batches = np.mean(np.array(self.persistent_loss_all_batches), axis=0)
        self.stochastic_loss_all_batches = np.mean(np.array(self.stochastic_loss_all_batches), axis=0)
        assert len(np.array(self.persistent_loss_all_batches).shape) == 1
        assert np.array(self.persistent_loss_all_batches).shape[0] == self.future_length
        print("Bug here:", np.array(self.stochastic_loss_all_batches).shape)
        assert len(np.array(self.stochastic_loss_all_batches).shape) == 2
        assert np.array(self.stochastic_loss_all_batches).shape[0] == self.num_stochastic_samples

    def run_deterministic(self):
        """
        Revised version by ML: more explicit (since not every thing is populated in class attributes),
                               but still inefficient!
        Loops over all samples of the test dataset to produce forecasts which are then saved to netCDF-files
        Besides, basic evaluation metrics are calculated and saved (see return)
        :return: Populated versions of self.stochastic_loss_all_batches and self.stochastic_loss_all_batches
        """
        method = Postprocess.run_deterministic.__name__

        # init the session and restore the trained model
        self.init_session()
        self.restore(self.sess, self.checkpoint)

        # init sample index for looping and acculmulators for evaulation metrics
        sample_ind = 0
        fcst_mse_all, fcst_psnr_all = [], []
        prst_mse_all, prst_psnr_all = [], []

        while sample_ind < self.num_samples_per_epoch:
            # get normalized and denormalized input data
            input_results, input_images_denorm_all, t_starts = self.get_input_data_per_batch()
            # feed and run the trained model
            feed_dict = {input_ph: input_results[name] for name, input_ph in self.inputs.items()}
            # returned array has the shape [batchsize, seq_len, lat, lon, channel]
            gen_images = self.sess.run(self.video_model.outputs['gen_images'], feed_dict=feed_dict)
            # The forecasted sequence length is smaller since the last one is not used for comparison with groud truth
            # ML: Isn't it the first?
            assert gen_images.shape[1] == self.sequence_length - 1, \
                "%{0}: Sequence length of prediction must be smaller by one than total sequence length.".format(method)

            for i in np.arange(self.batch_size):
                ts = Postprocess.generate_seq_timestamps(t_starts[i], len_seq=self.sequence_length)
                # get persistence forecast for sequences at hand
                persistence_images = self.get_persistence_forecast_per_sample(ts)
                # get model prediction
                gen_images_denorm = Postprocess.denorm_images_all_channels(self.stat_fl, gen_images[i], self.vars_in)
                # save sequences to netcdf-file and track initial time
                self.ts_fcst_ini.append(ts[self.context_frames])
                nc_fname = os.path.join(self.results_dir, "vfp_date_{0}_sample_ind_{1:d}.nc"
                                        .format(ts[self.context_frames].strftime("%Y%m%d%H"), sample_ind + i))
                print("%{0}: Save sequence data to nectCDF-file '{1}'".format(method, nc_fname))
                self.save_sequences_to_netcdf(input_images_denorm_all[i], persistence_images,
                                              np.expand_dims(np.array(gen_images_denorm), axis=0), ts, nc_fname)

                prst_mse_all.append(Postprocess.calculate_sample_metrics(input_images_denorm_all[i],
                                                                         persistence_images, self.future_length,
                                                                         self.context_frames, metric="mse", channel=0))
                prst_psnr_all.append(Postprocess.calculate_sample_metrics(input_images_denorm_all[i],
                                                                          persistence_images, self.future_length,
                                                                          self.context_frames, metric="psnr", channel=0))
                fcst_mse_all.append(Postprocess.calculate_sample_metrics(input_images_denorm_all[i],
                                                                         gen_images_denorm, self.future_length,
                                                                         self.context_frames, metric="mse", channel=0))
                fcst_psnr_all.append(Postprocess.calculate_sample_metrics(input_images_denorm_all[i],
                                                                          gen_images_denorm, self.future_length,
                                                                          self.context_frames, metric="psnr", channel=0))
                # end of batch-loop
            sample_ind += self.batch_size
            # end of while-loop for samples
        self.average_eval_metrics_for_all_batches(prst_mse_all, prst_psnr_all, fcst_mse_all, fcst_psnr_all)
        self.add_ensemble_dim()

    def calculate_persistence_eval_metrics(self, i):
        """
        Calculates MSE and PSNR for one persistence forecast
        :param i: index of persistence forecast sequence
        :return mse_sample: MSE-value
        :return psnr_sample: PSNR-value
        """
        # calculate the evaluation metric for persistent and model forecasting per sample
        mse_sample = Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i],
                                                                             self.persistence_images,
                                                                             self.future_length, self.context_frames,
                                                                             metric="mse", channel=0)

        psnr_sample = Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i],
                                                                              self.persistence_images,
                                                                              self.future_length,
                                                                              self.context_frames, metric="psnr",
                                                                              channel=0)

        return mse_sample, psnr_sample

    def calculate_forecast_eval_metrics(self, i):
        """
        Calculates MSE and PSNR for one model forecast
        :param i: index of model forecast sequence
        :return mse_sample: MSE-value
        :return psnr_sample: PSNR-value
        """
        mse_sample = Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i],
                                                                      self.gen_images_denorm, self.future_length,
                                                                      self.context_frames, metric="mse", channel=0)
        psnr_sample = Postprocess.calculate_metrics_by_sample(self.input_images_denorm_all[i],
                                                              self.gen_images_denorm, self.future_length,
                                                              self.context_frames, metric="psnr", channel=0)
        return mse_sample, psnr_sample

    def average_eval_metrics_for_all_batches(self, prst_mse_all, prst_psnr_all, fcst_mse_all, fcst_psnr_all):
        """
        Average evaluation metrics for all the samples.
        For all variables, the first dimension (axis=0) must be the training examples of the mini-batch
        :param prst_mse_all: MSE of all persistence forecasts
        :param prst_psnr_all: PSNR of all persistence forecasts
        :param fcst_mse_all: MSE of all model forecasts
        :param fcst_psnr_all: PSNR of all model forecasts
        """
        self.prst_metric_mse_all = np.mean(np.array(prst_mse_all), axis=0)
        self.prst_metric_psnr_all = np.mean(np.array(prst_psnr_all), axis=0)

        self.fcst_metric_mse_all = np.mean(np.array(fcst_mse_all), axis=0)
        self.fcst_metric_psnr_all = np.mean(np.array(fcst_psnr_all), axis=0)

    def add_ensemble_dim(self):
        """
        Expands dimensions of loss-arrays by dummy ensemble-dimension (used for deterministic forecasts only)
        :return:
        """
        self.stochastic_loss_all_batches = np.expand_dims(self.fcst_metric_mse_all, axis=0)  # [1,future_lenght]
        self.stochastic_loss_all_batches_psnr = np.expand_dims(self.fcst_metric_psnr_all, axis=0)  # [1,future_lenght]

    def get_persistence_forecast_per_sample(self, t_seq):
        """
        Function to retrieve persistence forecast for each sample
        :param t_seq: sequence of datetime objects for which persistent forecast should be retrieved
        """
        method = Postprocess.get_persistence_forecast_per_sample.__name__

        # ML: init_date_str and ts_persistence are redundant
        # self.init_date_str = self.ts[0].strftime("%Y%m%d%H")
        # persistence_images, self.ts_persistence = Postprocess.get_persistence(self.ts, self.input_dir_pkl)
        # get persistence_images
        persistence_images, _ = Postprocess.get_persistence(t_seq, self.input_dir_pkl)
        assert persistence_images.shape[0] == self.sequence_length - 1,\
            "%{0}: Unexpected sequence length of persistence forecast".format(method)

        # ML: Do not plot inside loop
        # self.plot_persistence_images()
        return persistence_images

    def run(self):
        if self.model == "convLSTM" or self.model == "test_model" or self.model == 'mcnet':
            self.run_deterministic()
        elif self.run_mode == "deterministic":
            self.run_deterministic()
        else:
            self.run_stochastic()

    @staticmethod
    def calculate_metrics_by_batch(input_per_batch, output_per_batch, future_length, context_frames, metric="mse",
                                   channel=0):
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
        assert len(output_per_batch.shape) == 5
        eval_metrics_by_ts = []
        for ts in range(future_length):
            if metric == "mse":
                loss = (np.square(input_per_batch[:, context_frames + ts, :, :, channel] - output_per_batch[:,
                                                                                           context_frames + ts - 1, :,
                                                                                           :, channel])).mean()
            eval_metrics_by_ts.append(loss)
        assert len(eval_metrics_by_ts) == future_length
        return eval_metrics_by_ts

    @staticmethod
    def calculate_sample_metrics(input_per_sample, output_per_sample, future_length, context_frames, metric, channel):

        method = Postprocess.calculate_sample_metrics.__name__

        input_per_sample = np.array(input_per_sample)
        output_per_sample = np.array(output_per_sample)
        eval_metrics_by_ts = []
        for ts in range(future_length):
            if metric == "mse":
                loss = (np.square(
                    input_per_sample[context_frames + ts, :, :, channel] - output_per_sample[context_frames + ts - 1, :,
                                                                           :, channel])).mean()
            elif metric == "psnr":
                loss = metrics.psnr_imgs(input_per_sample[context_frames + ts, :, :, channel],
                                         output_per_sample[context_frames + ts - 1, :, :, channel])
            else:
                raise ValueError("%{0}: Currently, only 'mse' and 'psnr' are supported for detereminstic forecasting"
                                 .format(method))
            eval_metrics_by_ts.append(loss)
        return eval_metrics_by_ts

    def save_one_eval_metric_to_json(self, metric="mse"):
        """
        save list to pickle file in results directory
        """
        self.eval_metrics = {}
        if metric == "mse":
            fcst_metric_all = self.fcst_metric_mse_all  # mse loss
            prst_metric_all = self.prst_metric_mse_all
        elif metric == "psnr":
            fcst_metric_all = self.fcst_metric_psnr_all  # psnr_loss
            prst_metric_all = self.prst_metric_psnr_all
        else:
            raise ValueError(
                "We currently only support metric 'mse' and  'psnr' as evaluation metric for detereminstic forecasting")
        for ts in range(self.future_length):
            self.eval_metrics["persistent_ts_" + str(ts)] = [str(prst_metric_all[ts])]
            # for stochastic_sample_ind in range(self.num_stochastic_samples):
            self.eval_metrics["model_ts_" + str(ts)] = [str(i) for i in fcst_metric_all[:, ts]]
        with open(os.path.join(self.results_dir, metric), "w") as fjs:
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
    def denorm_images(stat_fl, input_images_, channel, var):
        """
        denormaize one channel of images for particular var
        args:
            stat_fl       : str, the path of the statistical json file
            input_images_ : list/array [seq, lat,lon,channel], the input images are  denormalized
            channel       : the channel of images going to be denormalized
            var           : the variable name of the channel,

        """
        norm_cls = Norm_data(var)
        norm = 'minmax'  # TODO: can be replaced by loading option.json from previous step, if this information is saved there.
        with open(stat_fl) as js_file:
            norm_cls.check_and_set_norm(json.load(js_file), norm)
        input_images_denorm = norm_cls.denorm_var(input_images_[:, :, :, channel], var, norm)
        return input_images_denorm

    @staticmethod
    def denorm_images_all_channels(stat_fl, input_images_, vars_in):
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
            input_images_all_channles_denorm.append(Postprocess.denorm_images(stat_fl, input_images_,
                                                                              channel=c, var=vars_in[c]))
        input_images_denorm = np.stack(input_images_all_channles_denorm, axis=-1)
        return input_images_denorm

    @staticmethod
    def get_one_seq_from_batch(input_images, i):
        """
        Get one sequence images from batch images
        """
        assert (len(np.array(input_images).shape) == 5)
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
        if isinstance(t_start, int): t_start = str(t_start)
        if isinstance(t_start, np.ndarray):
            warnings.warn("You give array of timestamps, we only use the first timestamp as start datetime " +
                          "to generate sequence timestamps")
            t_start = str(t_start[0])
        if not len(t_start) == 10:
            raise ValueError("The timestamp gived should following the pattern '%Y%m%d%H' : 2017121209")
        s_datetime = datetime.datetime.strptime(t_start, '%Y%m%d%H')
        seq_ts = [s_datetime + datetime.timedelta(hours=i) for i in range(len_seq)]
        return seq_ts

    def save_sequences_to_netcdf(self, input_seq, persistence_seq, predicted_seq, ts, nc_fname):
        """
        Save the input images, persistent images and generated stochatsic images to netCDF file.
        Note that the seq-dimension must comprise the whole sequence length
        :param input_seq: sequence of input images [seq, lat, lon, channel]
        :param persistence_seq: sequence of images from persistence forecast [seq, lat, lon, channel]
        :param predicted_seq: sequence of forecasted images [stochastic_index ,seq, lat, lon, channel]
        :param ts: timestamp array of current sequence
        :param nc_fname: name of netCDF-file to be created
        :return None:
        """
        method = Postprocess.__name__

        # preparation: convert to NumPy-arrays and perform sanity checks
        input_seq, persistence_seq = np.asarray(input_seq), np.asarray(persistence_seq)
        predicted_seq = np.asarray(predicted_seq)

        in_seq_shape, per_seq_shape = np.shape(input_seq), np.shape(persistence_seq)
        pred_seq_shape = np.shape(predicted_seq)
        # sequence length of the prediction (2nd dimension) is smaller by one compared to input sequence
        pred_seq_shape_test = np.asarray(pred_seq_shape)
        pred_seq_shape_test[1] -= 1

        # further dimensions
        nlat, nlon = len(self.lats), len(self.lons)
        ntimes = len(ts)
        nvars = len(self.vars_in)

        # sanity checks
        assert in_seq_shape == per_seq_shape, "%{0}: Input sequence and persistence sequence must have the same shape." \
            .format(method)
        assert len(in_seq_shape) == 4, "%{0}: Number of dimensions of input and persistence sequence must be 4." \
            .format(method)
        assert in_seq_shape == tuple(pred_seq_shape_test), "%{0}: Dimension of input sequence does ".format(method) + \
                                                           "not match dimension of predicted sequence"

        assert ntimes == in_seq_shape[0], "%{0}: Unexpected sequence length of input data ({1:d} vs. {2:d})" \
            .format(method, ntimes, in_seq_shape[0])
        assert nlat == in_seq_shape[1], "%{0}: Unexpected number of data points in y-direction ({1:d} vs. {2:d})" \
            .format(method, nlat, in_seq_shape[1])
        assert nlon == in_seq_shape[2], "%{0}: Unexpected number of data points in x-direction ({1:d} vs. {2:d})" \
            .format(method, nlon, in_seq_shape[2])

        assert nvars == in_seq_shape[3], "%{0}: Unexpected number of channels ({1:d} vs. {2:d}".format(method, nvars,
                                                                                                       in_seq_shape[3])
        # create datasets
        attr_dict = {"title": "Input, persistence and forecast data created by model stored under {0}"
            .format(self.checkpoint),
                     "author": "AMBS team",
                     "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

        try:
            data_dict_input = dict([("{0}_input".format(self.vars_in[i]), (["time_input", "lat", "lon"],
                                    input_seq[:self.context_frames, :, :, i]))
                                    for i in np.arange(nvars)])

            ds_input = xr.Dataset(data_dict_input,
                                  coords={"time_input": ts[self.context_frames:],
                                          "lat": self.lats, "lon": self.lons},
                                  attrs=attr_dict)
        except Exception as err:
            print("%{0}: Something went wrong when creating dataset for input data.".format(method))
            raise err

        try:
            data_dict_fcst = dict([("{0}_fcst".format(self.vars_in[i]), (["time_forecast", "lat", "lon"],
                                   predicted_seq[0, self.context_frames - 1:, :, :, i]))
                                   for i in np.arange(nvars)])

            ds_forecast = xr.Dataset(data_dict_fcst,
                                     coords={"time_forecast": ts[self.context_frames:],
                                             "lat": self.lats, "lon": self.lons},
                                     attrs=attr_dict)
        except Exception as err:
            print("%{0}: Something went wrong when creating dataset for forecast data.".format(method))
            raise err

        try:
            data_dict_per = dict([("{0}_prst".format(self.vars_in[i]), (["time_forecast", "lat", "lon"],
                                  persistence_seq[self.context_frames:, :, :, i]))
                                  for i in np.arange(nvars)])

            ds_persistence = xr.Dataset(data_dict_per,
                                        coords={"time_forecast": ts[self.context_frames:],
                                                "lat": self.lats, "lon": self.lons},
                                        attrs=attr_dict)
        except Exception as err:
            print("%{0}: Something went wrong when creating dataset for persistence forecast data.".format(method))
            raise err

        encode_nc = {key: {"zlib": True, "complevel": 5} for key in list(ds_input.keys())}

        # populate data in netCDF-file (take care for the mode!)
        ds_input.create_netcdf(nc_fname, encoding=encode_nc)
        ds_persistence.create_netcdf(nc_fname, mode="a", encoding=encode_nc)
        ds_forecast.create_netcdf(nc_fname, mode="a", encoding=encode_nc)

        print("%{0}: Data-file {1} was created successfully".format(method, nc_fname))

        return None

    @staticmethod
    def get_persistence(ts, input_dir_pkl):
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
        for t in range(len(ts)):  # Scarlet: this certainly can be made nicer with list comprehension
            ts_temp = ts[t] - datetime.timedelta(days=1)
            ts_persistence.append(ts_temp)
        t_persistence_start = ts_persistence[0]
        t_persistence_end = ts_persistence[-1]
        year_start = t_persistence_start.year
        month_start = t_persistence_start.month
        month_end = t_persistence_end.month
        print("start year:", year_start)
        # only one pickle file is needed (all hours during the same month)
        if month_start == month_end:
            # Open files to search for the indizes of the corresponding time
            time_pickle = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'T'))
            # Open file to search for the correspoding meteorological fields
            var_pickle = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'X'))

            if year_origin != year_start:
                time_origin_pickle = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_origin, 12, 'T'))
                var_origin_pickle = list(Postprocess.load_pickle_for_persistence(input_dir_pkl, year_origin, 12, 'X'))
                time_pickle.extend(time_origin_pickle)
                var_pickle.extend(var_origin_pickle)

            # Retrieve starting index
            ind = list(time_pickle).index(np.array(ts_persistence[0]))
            # print('Scarlet, Original', ts_persistence)
            # print('From Pickle', time_pickle[ind:ind+len(ts_persistence)])

            var_persistence = np.array(var_pickle)[ind:ind + len(ts_persistence)]
            time_persistence = np.array(time_pickle)[ind:ind + len(ts_persistence)].ravel()
            # print(' Scarlet Shape of time persistence',time_persistence.shape)
            # print(' Scarlet Shape of var persistence',var_persistence.shape)


        # case that we need to derive the data from two pickle files (changing month during the forecast periode)
        else:
            t_persistence_first_m = []  # should hold dates of the first month
            t_persistence_second_m = []  # should hold dates of the second month

            for t in range(len(ts)):
                m = ts_persistence[t].month
                if m == month_start:
                    t_persistence_first_m.append(ts_persistence[t])
                if m == month_end:
                    t_persistence_second_m.append(ts_persistence[t])
            if year_origin == year_start:
                # Open files to search for the indizes of the corresponding time
                time_pickle_first = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'T')
                time_pickle_second = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_end, 'T')

                # Open file to search for the correspoding meteorological fields
                var_pickle_first = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_start, 'X')
                var_pickle_second = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, month_end, 'X')

            if year_origin != year_start:
                # Open files to search for the indizes of the corresponding time
                time_pickle_second = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_origin, 1, 'T')
                time_pickle_first = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, 12, 'T')

                # Open file to search for the correspoding meteorological fields
                var_pickle_second = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_origin, 1, 'X')
                var_pickle_first = Postprocess.load_pickle_for_persistence(input_dir_pkl, year_start, 12, 'X')

            # Retrieve starting index
            ind_first_m = list(time_pickle_first).index(np.array(t_persistence_first_m[0]))
            print("time_pickle_second:", time_pickle_second)
            ind_second_m = list(time_pickle_second).index(np.array(t_persistence_second_m[0]))

            # append the sequence of the second month to the first month
            var_persistence = np.concatenate((var_pickle_first[ind_first_m:ind_first_m + len(t_persistence_first_m)],
                                              var_pickle_second[
                                              ind_second_m:ind_second_m + len(t_persistence_second_m)]),
                                             axis=0)
            time_persistence = np.concatenate((time_pickle_first[ind_first_m:ind_first_m + len(t_persistence_first_m)],
                                               time_pickle_second[
                                               ind_second_m:ind_second_m + len(t_persistence_second_m)]),
                                              axis=0).ravel()  # ravel is needed to eliminate the unnecessary dimension (20,1) becomes (20,)
            # print(' Scarlet concatenate and ravel (time)', var_persistence.shape, time_persistence.shape)

        if len(time_persistence.tolist()) == 0: raise ("The time_persistent is empty!")
        if len(var_persistence) == 0: raise ("The var persistence is empty!")
        # tolist() is needed for plottingi
        var_persistence = var_persistence[1:]
        time_persistence = time_persistence[1:]
        return var_persistence, time_persistence.tolist()

    @staticmethod
    def load_pickle_for_persistence(input_dir_pkl, year_start, month_start, pkl_type):
        """Helper to get the content of the pickle files. There are two types in our workflow:
        T_[month].pkl where the time stamp is stored
        X_[month].pkl where the variables are stored, e.g. temperature, geopotential and pressure
        This helper function constructs the directory, opens the file to read it, returns the variable.
        """
        path_to_pickle = input_dir_pkl + '/' + str(year_start) + '/' + pkl_type + '_{:02}.pkl'.format(month_start)
        infile = open(path_to_pickle, 'rb')
        var = pickle.load(infile)
        return var

    def plot_evalution_metrics(self):
        model_names = self.eval_metrics.keys()
        model_ts_errors = []  # [timestamps,stochastic_number]
        persistent_ts_errors = []
        for ts in range(self.future_length - 1):
            stochastic_err = self.eval_metrics["model_ts_" + str(ts)]
            stochastic_err = [float(item) for item in stochastic_err]
            model_ts_errors.append(stochastic_err)
            persistent_err = self.eval_metrics["persistent_ts_" + str(ts)]
            persistent_err = float(persistent_err[0])
            persistent_ts_errors.append(persistent_err)
        if len(np.array(model_ts_errors).shape) == 1:
            model_ts_errors = np.expand_dims(np.array(model_ts_errors), axis=1)
        model_ts_errors = np.array(model_ts_errors)
        persistent_ts_errors = np.array(persistent_ts_errors)
        fig = plt.figure(figsize=(6, 4))
        ax = plt.axes([0.1, 0.15, 0.75, 0.75])
        for stoch_ind in range(len(model_ts_errors[0])):
            plt.plot(model_ts_errors[:, stoch_ind], lw=1)
        plt.plot(persistent_ts_errors)
        plt.xticks(np.arange(1, self.future_length))
        ax.set_ylim(0., 10)
        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Time stamps')
        ax.set_ylabel("Errors")
        print("Saving plot for err")
        plt.savefig(os.path.join(self.results_dir, "evaluation.png"))

    def plot_example_forecasts(self, metric="mse", var_ind=0):
        """
        Plots example forecasts. The forecasts are chosen from the complete pool of the test dataset and are chosen
        according to the accuracy in terms of the chosen metric. In add ition, to the best and worst forecast,
        every decil of the chosen metric is retrieved to cover the whole bandwith of forecasts.
        :param metric: The metric which is used for measuring accuracy
        :param var_ind: The index of the forecasted variable to plot (correspondong to self.vars_in)
        :return: 11 forecast plots are created
        """

        method = Postprocess.plot_example_forecasts.__name__

        quantiles = np.arange(0., 1.01, .1)

        metric_data, quantiles_val = Postprocess.get_quantiles(quantiles, metric)
        quantiles_inds = Postprocess.get_matching_indices(metric_data, quantiles_val)

        for i in quantiles_inds:
            date_curr = self.ts_fcst_ini[i]
            nc_fname = os.path.join(self.results_dir, "vfp_date_{0}_sample_ind_{1:d}.nc"
                                    .format(date_curr.strftime("%Y%m%d%H"), i))
            if not os.path.isfile(nc_fname):
                raise FileNotFoundError("%{0}: Could not find requested file '{1}'".format(method, nc_fname))
            else:
                # get the data
                varname = self.vars_in[var_ind]
                with xr.open_dataset(nc_fname) as dfile:
                    data_fcst = dfile["{0}_fcst".format(varname)]
                    data_ref = dfile["{0}_ref".format(varname)]

                data_diff = data_fcst - data_ref
                dates_fcst = pd.to_datetime(data_ref.coords["time_forecast"].data)
                # name of plot
                plt_fname_base = os.path.join(self.output_dir, "forecast_{0}_{1}_{2}_{3:d}percentile.png"
                                              .format(varname, dates_fcst[0].strftime("%Y%m%dT%H00"), metric,
                                                      int(quantiles[i]*100.)))

                self.create_plot(data_fcst, data_diff, varname, plt_fname_base)

    def get_quantiles(self, quantiles, metric="mse"):
        """
        Get the quantiles for the metric of interest.
        :param quantiles: The quantiles for which the index should be obtained
        :param metric: the metric of interest ("mse" and "psnr" are currently available)
        :return data: the array holding the metric of interst
        :return quantiles_vals: the requested quantile values
        """

        method = Postprocess.get_quantile_inds.__name__

        if metric == "mse":
            if self.fcst_metric_mse_all is None:
                raise ValueError("%{0}: fcst_metric_mse_all-attribute storing forecast MSE is still uninitialized."
                                 .format(method))
            data = np.array(self.fcst_metric_mse_all)

        elif metric == "psnr":
            if self.fcst_metric_psnr_all is None:
                raise ValueError("%{0}: fcst_metric_psnr_all-attribute storing forecast PSNR is still uninitialized."
                                 .format(method))
            data = np.array(self.fcst_metric_psnr_all)
        else:
            raise ValueError("%{0}: Metric {1} is unknown.".format(method, metric))

        quantiles_vals = np.quantile(np.array(data), quantiles, interpolation="nearest")

        return data, quantiles_vals


    @staticmethod
    def get_matching_indices(big_array, subset):
        """
        Returns the indices where element values match the values in an array
        :param big_array: the array to dig through
        :param subset: array of values contained in big_array
        :return: the desired indices
        """

        sorted_keys = np.argsort(big_array)
        indexes = sorted_keys[np.searchsorted(big_array, subset, sorter=sorted_keys)]

        return indexes

    @staticmethod
    def create_plot(data, data_diff, varname, plt_fname):
        """
        Creates filled contour plot of forecast data and also draws contours for differences.
        ML: So far, only plotting of the 2m temperature is supported (with 12 predicted hours/frames)
        :param data: the forecasted data array to be plotted
        :param data_diff: the reference data ('ground truth')
        :param varname: the name of the variable
        :param plt_fname: the filename to the store the plot
        :return: -
        """
        method = Postprocess.create_plot.__name__

        try:
            coords = data.coords
            # handle coordinates and forecast times
            lat, lon = coords["lat"], coords["lon"]
            dates_fcst = pd.to_datetime(coords["time_forecast"].data)
        except Exception as err:
            print("%{0}: Could not retrieve expected coordinates lat, lon and time_forecast from data.".format(method))
            raise err

        lons, lats = np.meshgrid(lon, lat)

        date0 = dates_fcst[0] - (dates_fcst[1] - dates_fcst[0])
        date0_str = date0.strftime("%Y-%m-%d %H:%M UTC")

        fhhs = (dates_fcst - date0) / pd.Timedelta('1 hour')

        # check data to be plotted since programme is not generic so far
        if np.shape(dates_fcst)[0] != 12:
            raise ValueError("%{0}: Currently, only 12 hour forecast can be handled properly.".format(method))

        if varname != "2t":
            raise ValueError("%{0}: Currently, only 2m temperature is plotted nicely properly.".format(method))

        # define levels
        clevs = np.arange(-10., 40., 1.)
        clevs_diff = np.arange(0.5, 10.5, 2.)
        clevs_diff2 = np.arange(-10.5, -0.5, 2.)

        # create fig and subplot axes
        fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(12, 6))
        axes = axes.flatten()

        # create all subplots
        for t, fhh in fhhs:
            m = Basemap(projection='cyl', llcrnrlat=np.min(lat), urcrnrlat=np.max(lat),
                        llcrnrlon=np.min(lon), urcrnrlon=np.max(lon), resolution='l', ax=axes[t])
            m.drawcoastlines()
            x, y = m(lons, lats)
            if t%6 == 0:
                lat_lab = [1,0,0,0]
                axes[t].set_ylabel(u'Latitude', labelpad=30)
            else:
                lat_lab = list(np.zeros(4))
            if t/6 >= 1:
                lon_lab = [0,0,0,1]
                axes[t].set_xlabel(u'Longitude', labelpad=15)
            else:
                lon_lab = list(np.zeros(4))
            m.drawmapboundary()
            m.drawparallels(np.arange(0,90,5),labels=lat_lab, xoffset=1.)
            m.drawmeridians(np.arange(5,355,10),labels=lon_lab, yoffset=1.)
            cs = m.contourf(x, y, data.isel(time_forecast=t)-273.15, clevs, cmap=plt.get_cmap("jet"), ax=axes[t])
            cs_c_pos = m.contour(x, y, data_diff.isel(time_forecast=t), clevs_diff, linewidths=0.5, ax=axes[t],
                                 colors="black")
            cs_c_neg = m.contour(x, y, data_diff.isel(time_forecast=t), clevs_diff2, linewidths=1, linestyles="dotted",
                                 ax=axes[t], colors="black")
            axes[t].set_title("{0} +{1:02d}:00".format(date0_str, int(fhh)), fontsize=7.5, pad=4)

        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=-0.7,
                            wspace=0.05)
        # add colorbar.
        cbar_ax = fig.add_axes([0.3, 0.22, 0.4, 0.02])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation="horizontal")
        cbar.set_label('°C')
        # save to disk
        plt.savefig(plt_fname, bbox_inches="tight")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default='results',
                        help="ignored if output_gif_dir is specified")
    parser.add_argument("--checkpoint",
                        help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test'], default='test',
                        help='mode for dataset, val or test.')
    parser.add_argument("--batch_size", type=int, default=8, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--num_stochastic_samples", type=int, default=1)
    parser.add_argument("--stochastic_plot_id", type=int, default=0,
                        help="The stochastic generate images index to plot")
    parser.add_argument("--gpu_mem_frac", type=float, default=0.95, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    # ML: test_instance is a bit misleading here
    test_instance = Postprocess(results_dir=args.results_dir, checkpoint=args.checkpoint, mode="test",
                                batch_size=args.batch_size, num_samples=args.num_samples,
                                num_stochastic_samples=args.num_stochastic_samples, gpu_mem_frac=args.gpu_mem_frac,
                                seed=args.seed, stochastic_plot_id=args.stochastic_plot_id, args=args)

    test_instance()
    test_instance.run()
    test_instance.save_eval_metric_to_json()
    test_instance.plot_evalution_metrics()
    test_instance.plot_example_forecasts(metric="mse")

if __name__ == '__main__':
    main()