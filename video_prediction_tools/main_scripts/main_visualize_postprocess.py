from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Yan Ji, Michael Langguth"
__date__ = "2020-11-10"

import argparse
import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
import pickle
import datetime as dt
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from normalization import Norm_data
from metadata import MetaData as MetaData
from main_scripts.main_train_models import *
from data_preprocess.preprocess_data_step2 import *
from model_modules.video_prediction import datasets, models, metrics
from statistical_evaluation import perform_block_bootstrap_metric, avg_metrics


class Postprocess(TrainModel):
    def __init__(self, results_dir=None, checkpoint=None, mode="test", batch_size=None, num_stochastic_samples=1,
                 stochastic_plot_id=0, gpu_mem_frac=None, seed=None, args=None, run_mode="deterministic"):
        """
        The function for inference, generate results and images
        results_dir   :str, The output directory to save results
        checkpoint    :str, The directory point to the checkpoints
        mode          :str, Default is test, could be "train","val", and "test"
        batch_size    :int, The batch size used for generating test samples for each iteration
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
        # forecast products and evaluation metrics to be handled in postprocessing
        self.eval_metrics = ["mse", "psnr"]
        self.fcst_products = {"persistence": "pfcst", "model": "mfcst"}
        # initialize dataset to track evaluation metrics and configure bootstrapping procedure
        self.eval_metrics_ds = None
        self.nboots_block = 1000
        self.block_length = 7 * 24    # this corresponds to a block length of 7 days when forecasts are produced every hour
        # other attributes
        self.stat_fl = None
        self.norm_cls = None            # placeholder for normalization instance
        self.channel = 0                # index of channel/input variable to evaluate
        self.num_samples_per_epoch = None
        # set further attributes from parsed arguments
        self.results_dir = self.output_dir = os.path.normpath(results_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.batch_size = batch_size
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.num_stochastic_samples = num_stochastic_samples
        self.stochastic_plot_id = stochastic_plot_id
        self.args = args
        self.checkpoint = checkpoint
        self.run_mode = run_mode
        self.mode = mode
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

    # methods that are executed with __call__
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
                options_checkpoint = json.loads(f.read())
                self.dataset = options_checkpoint["dataset"]
                self.model = options_checkpoint["model"]
                self.input_dir_tfr = options_checkpoint["input_dir"]
                self.input_dir = os.path.dirname(self.input_dir_tfr.rstrip("/"))
                self.input_dir_pkl = os.path.join(self.input_dir, "pickle")
                # update self.fcst_products
                if "model" in self.fcst_products.keys():
                    self.fcst_products[self.model] = self.fcst_products.pop("model")
        except Exception as err:
            print("%{0}: Something went wrong when reading the checkpoint-file '{1}'".format(method_name,
                                                                                             checkpoint_opt_dict))
            raise err

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

        try:
            md_instance = MetaData(json_file=metadata_fl)
        except Exception as err:
            print("%{0}: Something went wrong when getting metadata from file '{1}'".format(method_name, metadata_fl))
            raise err

        # when the metadat is loaded without problems, the follwoing will work
        self.height, self.width = md_instance.ny, md_instance.nx
        self.vars_in = md_instance.variables

        self.lats = xr.DataArray(md_instance.lat, coords={"lat": md_instance.lat}, dims="lat",
                                     attrs={"units": "degrees_east"})
        self.lons = xr.DataArray(md_instance.lon, coords={"lon": md_instance.lon}, dims="lon",
                                     attrs={"units": "degrees_north"})

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
        method = Postprocess.setup_num_samples_per_epoch.__name__

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
        test_tf_dataset = self.test_dataset.make_dataset(self.batch_size)
        test_iterator = test_tf_dataset.make_one_shot_iterator()
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        test_handle = test_iterator.string_handle()
        dataset_iterator = tf.data.Iterator.from_string_handle(test_handle, test_tf_dataset.output_types,
                                                               test_tf_dataset.output_shapes)
        self.inputs = dataset_iterator.get_next()
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

    # the run-factory
    def run(self):
        if self.model == "convLSTM" or self.model == "test_model" or self.model == 'mcnet':
            self.run_deterministic()
        elif self.run_mode == "deterministic":
            self.run_deterministic()
        else:
            self.run_stochastic()

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
                                                                   fl_name="vfp_date_{}_sample_ind_{}.nc"
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
        Revised and vectorized version of run_deterministic
        Loops over the training data, generates forecasts and calculates basic evaluation metrics on-the-fly
        """
        method = Postprocess.run_deterministic.__name__

        # init the session and restore the trained model
        self.init_session()
        self.restore(self.sess, self.checkpoint)

        # init sample index for looping and acculmulators for evaulation metrics
        sample_ind = 0
        nsamples = self.num_samples_per_epoch
        # initialize datasets
        eval_metric_ds = Postprocess.init_metric_ds(self.fcst_products, self.eval_metrics, self.vars_in[self.channel],
                                                    nsamples, self.future_length)

        while sample_ind < self.num_samples_per_epoch:
            # get normalized and denormalized input data
            input_results, input_images_denorm, t_starts = self.get_input_data_per_batch(self.inputs)
            # feed and run the trained model; returned array has the shape [batchsize, seq_len, lat, lon, channel]
            feed_dict = {input_ph: input_results[name] for name, input_ph in self.inputs.items()}
            gen_images = self.sess.run(self.video_model.outputs['gen_images'], feed_dict=feed_dict)
            # sanity check on length of forecast sequence
            assert gen_images.shape[1] == self.sequence_length - 1, \
                "%{0}: Sequence length of prediction must be smaller by one than total sequence length.".format(method)
            # denormalize forecast sequence (self.norm_cls is already set in get_input_data_per_batch-method)
            gen_images_denorm = self.denorm_images_all_channels(gen_images, self.vars_in, self.norm_cls,
                                                                norm_method="minmax")
            # store data into datset
            times_0, init_times = self.get_init_time(t_starts)
            batch_ds = self.create_dataset(input_images_denorm, gen_images_denorm, init_times)
            # auxilary list of forecast dimensions
            dims_fcst = list(batch_ds["{0}_ref".format(self.vars_in[0])].dims)

            for i in np.arange(self.batch_size):
                # work-around to make use of get_persistence_forecast_per_sample-method
                times_seq = (pd.date_range(times_0[i], periods=int(self.sequence_length), freq="h")).to_pydatetime()
                # get persistence forecast for sequences at hand and write to dataset
                persistence_seq, _ = Postprocess.get_persistence(times_seq, self.input_dir_pkl)
                for ivar, var in enumerate(self.vars_in):
                    batch_ds["{0}_persistence_fcst".format(var)].loc[dict(init_time=init_times[i])] = \
                        persistence_seq[self.context_frames-1:, :, :, ivar]

                # save sequences to netcdf-file and track initial time
                nc_fname = os.path.join(self.results_dir, "vfp_date_{0}_sample_ind_{1:d}.nc"
                                        .format(pd.to_datetime(init_times[i]).strftime("%Y%m%d%H"), sample_ind + i))
                self.save_ds_to_netcdf(batch_ds.isel(init_time=i), nc_fname)
                # end of batch-loop
            # write evaluation metric to corresponding dataset...
            eval_metric_ds = self.populate_eval_metric_ds(eval_metric_ds, batch_ds, sample_ind,
                                                          self.vars_in[self.channel])
            # ... and increment sample_ind
            sample_ind += self.batch_size
            # end of while-loop for samples
        # change init_time-coordinates to datetime64-type and safe dataset with evaluation metrics for later use
        eval_metric_ds = eval_metric_ds.assign_coords(init_time=pd.to_datetime(eval_metric_ds["init_time"]))
        self.eval_metrics_ds = eval_metric_ds
        #self.add_ensemble_dim()

    # all methods of the run factory
    def get_input_data_per_batch(self, input_iter, norm_method="minmax"):
        """
        Get the input sequence from the dataset iterator object stored in self.inputs and denormalize the data
        :param input_iter: the iterator object built by make_test_dataset_iterator-method
        :param norm_method: normalization method applicable to the data
        :return input_results: the normalized input data
        :return input_images_denorm: the denormalized input data
        :return t_starts: the initial time of the sequences
        """
        method = Postprocess.get_input_data_per_batch.__name__

        input_results = self.sess.run(input_iter)
        input_images = input_results["images"]
        t_starts = input_results["T_start"]
        if self.norm_cls is None:
            if self.stat_fl is None:
                raise AttributeError("%{0}: Attribute stat_fl is not initialized yet.".format(method))
            self.norm_cls = Postprocess.get_norm(self.vars_in, self.stat_fl, norm_method)

        # sanity check on input sequence
        assert np.ndim(input_images) == 5, "%{0}: Input sequence of mini-batch does not have five dimensions."\
                                           .format(method)

        input_images_denorm = Postprocess.denorm_images_all_channels(input_images, self.vars_in, self.norm_cls,
                                                                     norm_method=norm_method)

        return input_results, input_images_denorm, t_starts

    def get_init_time(self, t_starts):
        """
        Retrieves initial dates of forecast sequences from start time of whole inpt sequence
        :param t_starts: list/array of start times of input sequence
        :return: list of initial dates of forecast as numpy.datetime64 instances
        """
        method = Postprocess.get_init_time.__name__

        t_starts = np.squeeze(np.asarray(t_starts))
        if not np.ndim(t_starts) == 1:
            raise ValueError("%{0}: Inputted t_starts must be a 1D list/array of date-strings with format %Y%m%d%H"
                             .format(method))
        for i, t_start in enumerate(t_starts):
            try:
                seq_ts = pd.date_range(dt.datetime.strptime(str(t_start), "%Y%m%d%H"), periods=self.context_frames,
                                       freq="h")
            except Exception as err:
                print("%{0}: Could not convert {1} to datetime object. Ensure that the date-string format is 'Y%m%d%H'".
                      format(method, str(t_start)))
                raise err
            if i == 0:
                ts_all = np.expand_dims(seq_ts, axis=0)
            else:
                ts_all = np.vstack((ts_all, seq_ts))

        init_times = ts_all[:, -1]
        times0 = ts_all[:, 0]

        return times0, init_times

    def populate_eval_metric_ds(self, metric_ds, data_ds, ind_start, varname):
        """
        Populates evaluation metric dataset with values
        :param metric_ds: the evaluation metric dataset with variables such as 'mfcst_mse' (MSE of model forecast)
        :param data_ds: dataset holding the data from one mini-batch (see create_dataset-method)
        :param ind_start: start index of dimension init_time (part of metric_ds)
        :param varname: variable of interest (must be part of self.vars_in)
        :return: metric_ds
        """
        method = Postprocess.populate_eval_metric_ds.__name__

        # dictionary of implemented evaluation metrics
        known_eval_metrics = {"mse": Postprocess.calc_mse_batch , "psnr": Postprocess.calc_psnr_batch}

        # generate list of functions that calculate requested evaluation metrics
        if set(self.eval_metrics).issubset(known_eval_metrics):
            eval_metrics_func = [known_eval_metrics[metric] for metric in self.eval_metrics]
        else:
            misses = list(set(self.eval_metrics) - known_eval_metrics.keys())
            raise NotImplementedError("%{0}: The following requested evaluation metrics are not implemented yet: "
                                      .format(method, ", ".join(misses)))

        varname_ref = "{0}_ref".format(varname)
        init_times_metric = metric_ds["init_time"]
        it = init_times_metric[ind_start:ind_start+self.batch_size]
        for fcst_prod in self.fcst_products.keys():
            for imetric, eval_metric in enumerate(self.eval_metrics):
                metric_name = "{0}_{1}_{2}".format(varname, fcst_prod, eval_metric)
                varname_fcst = "{0}_{1}_fcst".format(varname, fcst_prod)
                metric_ds[metric_name].loc[dict(init_time=it)] = eval_metrics_func[imetric](data_ds[varname_fcst],
                                                                                            data_ds[varname_ref])
            # end of metric-loop
        # end of forecast product-loop
        # set init-time coordinate in place
        init_times_metric = init_times_metric.values
        init_times_metric[ind_start:ind_start+self.batch_size] = data_ds["init_time"]
        metric_ds = metric_ds.assign_coords(init_time=init_times_metric)
        return metric_ds

    def add_ensemble_dim(self):
        """
        Expands dimensions of loss-arrays by dummy ensemble-dimension (used for deterministic forecasts only)
        :return:
        """
        self.stochastic_loss_all_batches = np.expand_dims(self.fcst_mse_avg_batches, axis=0)  # [1,future_lenght]
        self.stochastic_loss_all_batches_psnr = np.expand_dims(self.fcst_psnr_avg_batches, axis=0)  # [1,future_lenght]

    def create_dataset(self, input_seq, fcst_seq, ts_ini):
        """
        Put input and forecast sequences into a xarray dataset. The latter also involves the persistence forecast
        which is just initialized, but unpopulated at this stage.
        The input data sequence is split into (effective) input sequence used for the forecast and into reference part.
        :param input_seq: sequence of input images [batch ,seq, lat, lon, channel]
        :param fcst_seq: sequence of forecast images [batch ,seq-1, lat, lon, channel]
        :param ts_ini: initial time of forecast (=last time step of effective input sequence)
        :return data_ds: above mentioned data in a nicely formatted dataset
        """

        method = Postprocess.create_dataset.__name__

        # auxiliary variables for temporal dimensions
        seq_hours = np.arange(self.sequence_length) - (self.context_frames-1)
        # some sanity checks
        assert np.shape(ts_ini)[0] == self.batch_size,\
            "%{0}: Inconsistent number of sequence start times ({1:d}) and batch size ({2:d})"\
            .format(method, np.shape(ts_ini)[0], self.batch_size)

        # turn input and forecast sequences to Data Arrays to ease indexing
        try:
            input_seq = xr.DataArray(input_seq, coords={"init_time": ts_ini, "fcst_hour": seq_hours,
                                                        "lat": self.lats, "lon": self.lons, "varname": self.vars_in},
                                     dims=["init_time", "fcst_hour", "lat", "lon", "varname"])
        except Exception as err:
            print("%{0}: Could not create Data Array for input sequence.".format(method))
            raise err

        try:
            fcst_seq = xr.DataArray(fcst_seq, coords={"init_time": ts_ini, "fcst_hour": seq_hours[1::],
                                                      "lat": self.lats, "lon": self.lons, "varname": self.vars_in},
                                    dims=["init_time", "fcst_hour", "lat", "lon", "varname"])
        except Exception as err:
            print("%{0}: Could not create Data Array for forecast sequence.".format(method))
            raise err

        # Now create the dataset where the input sequence is splitted into input that served for creating the
        # forecast and into the the reference sequences (which can be compared to the forecast)
        # as where the persistence forecast is containing NaNs (must be generated later)
        data_in_dict = dict([("{0}_in".format(var), input_seq.isel(fcst_hour=slice(None, self.context_frames),
                                                                   varname=ivar) \
                                                             .rename({"fcst_hour": "in_hour"})
                                                             .reset_coords(names="varname", drop=True))
                             for ivar, var in enumerate(self.vars_in)])

        # get shape of forecast data (one variable) -> required to initialize persistence forecast data
        shape_fcst = np.shape(fcst_seq.isel(fcst_hour=slice(self.context_frames-1, None), varname=0)
                                      .reset_coords(names="varname", drop=True))
        data_ref_dict = dict([("{0}_ref".format(var), input_seq.isel(fcst_hour=slice(self.context_frames, None),
                                                                     varname=ivar)
                                                               .reset_coords(names="varname", drop=True))
                              for ivar, var in enumerate(self.vars_in)])

        data_mfcst_dict = dict([("{0}_{1}_fcst".format(var, self.model),
                                 fcst_seq.isel(fcst_hour=slice(self.context_frames-1, None), varname=ivar)
                                         .reset_coords(names="varname", drop=True))
                                for ivar, var in enumerate(self.vars_in)])

        # fill persistence forecast variables with dummy data (to be populated later)
        data_pfcst_dict = dict([("{0}_persistence_fcst".format(var), (["init_time", "fcst_hour", "lat", "lon"],
                                                                       np.full(shape_fcst, np.nan)))
                                for ivar, var in enumerate(self.vars_in)])

        # create the dataset
        data_ds = xr.Dataset({**data_in_dict, **data_ref_dict, **data_mfcst_dict, **data_pfcst_dict})

        return data_ds

    def handle_eval_metrics(self):
        """
        Plots error-metrics averaged over all predictions to file.
        :return: a bunch of plots as png-files
        """
        method = Postprocess.handle_eval_metrics.__name__

        if self.eval_metrics_ds is None:
            raise AttributeError("%{0}: Attribute with dataset of evaluation metrics is still None.".format(method))

        # perform bootstrapping on metric dataset
        eval_metric_boot_ds = perform_block_bootstrap_metric(self.eval_metrics_ds, "init_time", self.block_length,
                                                             self.nboots_block)
        # ... and merge into existing metric dataset
        self.eval_metrics_ds = xr.merge([self.eval_metrics_ds, eval_metric_boot_ds])

        # calculate (unbootstrapped) averaged metrics
        eval_metric_avg_ds = avg_metrics(self.eval_metrics_ds, "init_time")
        # ... and merge into existing metric dataset
        self.eval_metrics_ds = xr.merge([self.eval_metrics_ds, eval_metric_avg_ds])

        # save evaluation metrics to file
        nc_fname = os.path.join(self.results_dir, "evaluation_metrics.nc")
        Postprocess.save_ds_to_netcdf(self.eval_metrics_ds, nc_fname)

        # also save averaged metrics to JSON-file and plot it for diagnosis
        _ = Postprocess.plot_avg_eval_metrics(self.eval_metrics_ds, self.eval_metrics, self.fcst_products,
                                              self.vars_in[self.channel], self.results_dir)

    # auxiliary methods (not necessarily bound to class instance)
    @staticmethod
    def get_norm(varnames, stat_fl, norm_method):
        """
        Retrieves normalization instance
        :param varnames: list of variabe names
        :param stat_fl: statistics JSON-file
        :param norm_method: normalization method
        :return: normalization instance which can be used to normalize images according to norm_method
        """
        method = Postprocess.get_norm.__name__

        if not isinstance(varnames, list):
            raise ValueError("%{0}: varnames must be a list of variable names.".format(method))

        norm_cls = Norm_data(varnames)
        try:
            with open(stat_fl) as js_file:
                norm_cls.check_and_set_norm(json.load(js_file), norm_method)
            norm_cls = norm_cls
        except Exception as err:
            print("%{0}: Could not handle statistics json-file '{1}'.".format(method, stat_fl))
            raise err
        return norm_cls

    @staticmethod
    def denorm_images_all_channels(image_sequence, varnames, norm, norm_method="minmax"):
        """
        Denormalize data of all image channels
        :param image_sequence: list/array [batch, seq, lat, lon, channel] of images
        :param varnames: list of variable names whose order matches channel indices
        :param norm: normalization instance
        :param norm_method: normalization-method (default: 'minmax')
        :return: denormalized image data
        """
        method = Postprocess.denorm_images_all_channels.__name__

        nvars = len(varnames)
        image_sequence = np.array(image_sequence)
        # sanity checks
        if not isinstance(norm, Norm_data):
            raise ValueError("%{0}: norm must be a normalization instance.".format(method))

        if nvars != np.shape(image_sequence)[-1]:
            raise ValueError("%{0}: Number of passed variable names ({1:d}) does not match number of channels ({2:d})"
                             .format(method, nvars, np.shape(image_sequence)[-1]))

        input_images_all_channles_denorm = [Postprocess.denorm_images(image_sequence, norm, {varname: c},
                                                                      norm_method=norm_method)
                                            for c, varname in enumerate(varnames)]

        input_images_denorm = np.stack(input_images_all_channles_denorm, axis=-1)
        return input_images_denorm

    @staticmethod
    def denorm_images(input_images, norm, var_dict, norm_method="minmax"):
        """
        Denormalize one channel of images
        :param input_images: list/array [batch, seq, lat, lon, channel]
        :param norm: normalization instance
        :param var_dict: dictionary with one key only mapping variable name to channel index, e.g. {"2_t": 0}
        :param norm_method: normalization method (default: minmax-normalization)
        :return: denormalized image data
        """
        method = Postprocess.denorm_images.__name__
        # sanity checks
        if not isinstance(var_dict, dict):
            raise ValueError("%{0}: var_dict is not a dictionary.".format(method))
        else:
            if len(var_dict.keys()) > 1:
                raise ValueError("%{0}: var_dict must contain one key only.".format(method))
            varname, channel = *var_dict.keys(), *var_dict.values()

        if not isinstance(norm, Norm_data):
            raise ValueError("%{0}: norm must be a normalization instance.".format(method))

        try:
            input_images_denorm = norm.denorm_var(input_images[..., channel], varname, norm_method)
        except Exception as err:
            print("%{0}: Something went wrong when denormalizing image sequence. Inspect error-message!".format(method))
            raise err

        return input_images_denorm

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
    def get_persistence(ts, input_dir_pkl):
        """
        This function gets the persistence forecast.
        'Today's weather will be like yesterday's weather.'
        :param ts: list dontaining datetime objects from get_init_times
        :param input_dir_pkl: input directory to pickle files
        :return time_persistence: list containing the dates and times of the persistence forecast.
        :return var_peristence: sequence of images corresponding to these times
        """
        ts_persistence = []
        year_origin = ts[0].year
        for t in range(len(ts)):  # Scarlet: this certainly can be made nicer with list comprehension
            ts_temp = ts[t] - dt.timedelta(days=1)
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

            var_persistence = np.array(var_pickle)[ind:ind + len(ts_persistence)]
            time_persistence = np.array(time_pickle)[ind:ind + len(ts_persistence)].ravel()
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
            #print("time_pickle_second:", time_pickle_second)
            ind_second_m = list(time_pickle_second).index(np.array(t_persistence_second_m[0]))

            # append the sequence of the second month to the first month
            var_persistence = np.concatenate((var_pickle_first[ind_first_m:ind_first_m + len(t_persistence_first_m)],
                                              var_pickle_second[
                                              ind_second_m:ind_second_m + len(t_persistence_second_m)]),
                                             axis=0)
            time_persistence = np.concatenate((time_pickle_first[ind_first_m:ind_first_m + len(t_persistence_first_m)],
                                               time_pickle_second[
                                               ind_second_m:ind_second_m + len(t_persistence_second_m)]),
                                              axis=0).ravel()
            # Note: ravel is needed to eliminate the unnecessary dimension (20,1) becomes (20,)

        if len(time_persistence.tolist()) == 0:
            raise ValueError("The time_persistent is empty!")
        if len(var_persistence) == 0:
            raise ValueError("The var persistence is empty!")

        var_persistence = var_persistence[1:]
        time_persistence = time_persistence[1:]

        return var_persistence, time_persistence.tolist()

    @staticmethod
    def load_pickle_for_persistence(input_dir_pkl, year_start, month_start, pkl_type):
        """
        There are two types in our workflow: T_[month].pkl where the timestamp is stored,
        X_[month].pkl where the variables are stored, e.g. temperature, geopotential and pressure.
        This helper function constructs the directory, opens the file to read it, returns the variable.
        :param input_dir_pkl: directory where input pickle files are stored
        :param year_start: The year for which data is requested as integer
        :param month_start: The year for which data is requested as integer
        :param pkl_type: Either "X" or "T"
        """
        path_to_pickle = os.path.join(input_dir_pkl, str(year_start), pkl_type + "_{:02}.pkl".format(month_start))
        with open(path_to_pickle, "rb") as pkl_file:
            var = pickle.load(pkl_file)
        return var

    @staticmethod
    def save_ds_to_netcdf(ds, nc_fname, comp_level=5):
        """
        Writes xarray dataset into netCDF-file
        :param ds: The dataset to be written
        :param nc_fname: Path and name of the target netCDF-file
        :param comp_level: compression level, must be an integer between 1 and 9 (defualt: 5)
        :return: -
        """
        method = Postprocess.save_ds_to_netcdf.__name__

        # sanity checks
        if not isinstance(ds, xr.Dataset):
            raise ValueError("%{0}: Argument 'ds' must be a xarray dataset.".format(method))

        if not isinstance(comp_level, int):
            raise ValueError("%{0}: Argument 'comp_level' must be an integer.".format(method))
        else:
            if comp_level < 1 or comp_level > 9:
                raise ValueError("%{0}: Argument 'comp_level' must be an integer between 1 and 9.".format(method))

        if not os.path.isdir(os.path.dirname(nc_fname)):
            raise NotADirectoryError("%{0}: The directory to store the netCDf-file does not exist.".format(method))

        encode_nc = {key: {"zlib": True, "complevel": comp_level} for key in ds.keys()}

        # populate data in netCDF-file (take care for the mode!)
        try:
            ds.to_netcdf(nc_fname, encoding=encode_nc)
            print("%{0}: netCDF-file '{1}' was created successfully.".format(method, nc_fname))
        except Exception as err:
            print("%{0}: Something unexpected happened when creating netCDF-file '1'".format(method, nc_fname))
            raise err

    @staticmethod
    def calc_mse_batch(data_fcst, data_ref):
        """
        Calculate mse of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged mse for each batch example
        """
        method = Postprocess.calc_mse_batch.__name__

        dims = data_fcst.dims
        # sanity checks
        if dims[0] != "init_time":
            raise ValueError("%{0}: First dimension of data must be init_time.".format(method))

        if not list(data_fcst.coords) == list(data_ref.coords):
            raise ValueError("%{0}: Input data arrays must have the same shape and coordinates.".format(method))

        mse = np.square(data_fcst - data_ref).mean(dim=["lat", "lon"])

        return mse.values

    @staticmethod
    def calc_psnr_batch(data_fcst, data_ref):
        """
        Calculate mse of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged mse for each batch example
        """
        method = Postprocess.calc_mse_batch.__name__

        dims = data_fcst.dims
        # sanity checks
        if dims[0] != "init_time":
            raise ValueError("%{0}: First dimension of data must be init_time.".format(method))

        if not list(data_fcst.coords) == list(data_ref.coords):
            raise ValueError("%{0}: Input data arrays must have the same shape and coordinates.".format(method))

        psnr = metrics.psnr_imgs(data_ref.values, data_fcst.values)

        return psnr

    @staticmethod
    def plot_avg_eval_metrics(eval_ds, eval_metrics, fcst_prod_dict, varname, out_dir):
        """
        Plots error-metrics averaged over all predictions to file incl. 90%-confidence interval that is estimated by
        block bootstrapping.
        :param eval_ds: The dataset storing all evaluation metrics for each forecast (produced by init_metric_ds-method)
        :param eval_metrics: list of evaluation metrics
        :param fcst_prod_dict: dictionary of forecast products, e.g. {"pfcst": "persistence forecast"}
        :param varname: the variable name for which the evaluation metrics are available
        :param out_dir: output directory to save the lots
        :return: a bunch of plots as png-files
        """
        method = Postprocess.plot_avg_eval_metrics.__name__

        # settings for block bootstrapping
        # sanity checks
        if not isinstance(eval_ds, xr.Dataset):
            raise ValueError("%{0}: Argument 'eval_ds' must be a xarray dataset.".format(method))

        if not isinstance(fcst_prod_dict, dict):
            raise ValueError("%{0}: Argument 'fcst_prod_dict' must be dictionary with short names of forecast product" +
                             "as key and long names as value.".format(method))

        try:
            nhours = np.shape(eval_ds.coords["fcst_hour"])[0]
        except Exception as err:
            print("%{0}: Input argument 'eval_ds' appears to be unproper.".format(method))
            raise err

        nmodels = len(fcst_prod_dict.values())
        colors = ["blue", "red", "black", "grey"]
        for metric in eval_metrics:
            # create a new figure object
            fig = plt.figure(figsize=(6, 4))
            ax = plt.axes([0.1, 0.15, 0.75, 0.75])
            hours = np.arange(1, nhours+1)

            metric2plt = np.full((nmodels, nhours), np.nan)
            metric2plt_max, metric2plt_min = metric2plt.copy(), metric2plt.copy()
            for ifcst, fcst_prod in enumerate(fcst_prod_dict.keys()):
                metric_name = "{0}_{1}_{2}_avg".format(varname, fcst_prod, metric)
                try:
                    metric2plt = eval_ds[metric_name]
                    metric_boot = eval_ds[metric_name+"_boot"]
                except Exception as err:
                    print("%{0}: Could not retrieve {1} and/or {2} from evaluation metric dataset."
                          .format(method, metric_name, metric_name+"_boot"))
                    raise err
                # plot the data
                metric2plt_min = metric_boot.quantile(0.05, dim="iboot")
                metric2plt_max = metric_boot.quantile(0.95, dim="iboot")
                plt.plot(hours, metric2plt, label=fcst_prod, color=colors[ifcst], marker="o")
                plt.fill_between(hours, metric2plt_min, metric2plt_max, facecolor=colors[ifcst], alpha=0.3)
            # configure plot
            plt.xticks(hours)
            ax.set_ylim(0., None)
            if metric == "psnr": ax.set_ylim(None, None)
            legend = ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
            ax.set_xlabel("Lead time [hours]")
            ax.set_ylabel(metric.upper())
            plt_fname = os.path.join(out_dir, "evaluation_{0}".format(metric))
            print("Saving basic evaluation plot in terms of {1} to '{2}'".format(method, metric, plt_fname))
            plt.savefig(plt_fname)

        plt.close()

        return True

    def plot_example_forecasts(self, metric="mse", channel=0):
        """
        Plots example forecasts. The forecasts are chosen from the complete pool of the test dataset and are chosen
        according to the accuracy in terms of the chosen metric. In add ition, to the best and worst forecast,
        every decil of the chosen metric is retrieved to cover the whole bandwith of forecasts.
        :param metric: The metric which is used for measuring accuracy
        :param channel: The channel index of the forecasted variable to plot (correspondong to self.vars_in)
        :return: 11 exemplary forecast plots are created
        """
        method = Postprocess.plot_example_forecasts.__name__
        
        metric_name = "{0}_{1}_{2}".format(self.vars_in[channel], self.model, metric)
        if not metric_name in self.eval_metrics_ds:
            raise ValueError("%{0}: Cannot find requested evaluation metric '{1}'".format(method, metric_name) +
                             " onto which selection of plotted forecast is done.")
        # average metric of interest and obtain quantiles incl. indices
        metric_mean = self.eval_metrics_ds[metric_name].mean(dim="fcst_hour")
        quantiles = np.arange(0., 1.01, .1)
        quantiles_val = metric_mean.quantile(quantiles, interpolation="nearest")
        quantiles_inds = self.get_matching_indices(metric_mean.values, quantiles_val)
        print(metric_mean.coords["init_time"])
        for i, ifcst in enumerate(quantiles_inds):
            date_init = pd.to_datetime(metric_mean.coords["init_time"][ifcst].data)
            nc_fname = os.path.join(self.results_dir, "vfp_date_{0}_sample_ind_{1:d}.nc"
                                    .format(date_init.strftime("%Y%m%d%H"), ifcst))
            if not os.path.isfile(nc_fname):
                raise FileNotFoundError("%{0}: Could not find requested file '{1}'".format(method, nc_fname))
            else:
                # get the data
                varname = self.vars_in[channel]
                with xr.open_dataset(nc_fname) as dfile:
                    data_fcst = dfile["{0}_{1}_fcst".format(varname, self.model)]
                    data_ref = dfile["{0}_ref".format(varname)]

                data_diff = data_fcst - data_ref
                # name of plot
                plt_fname_base = os.path.join(self.output_dir, "forecast_{0}_{1}_{2}_{3:d}percentile.png"
                                              .format(varname, date_init.strftime("%Y%m%dT%H00"), metric,
                                                      int(quantiles[i]*100.)))

                Postprocess.create_plot(data_fcst, data_diff, varname, plt_fname_base)

    @staticmethod
    def init_metric_ds(fcst_products, eval_metrics, varname, nsamples, nlead_steps):
        """
        Initializes dataset for storing evaluation metrics
        :param fcst_products: list of forecast products to be evaluated
        :param eval_metrics: list of forecast metrics to be calculated
        :param varname: name of the variable for which metrics are calculated
        :param nsamples: total number of forecast samples
        :param nlead_steps: number of forecast steps
        :return: eval_metric_ds
        """
        eval_metric_dict = dict([("{0}_{1}_{2}".format(varname ,*(fcst_prod, eval_met)), (["init_time", "fcst_hour"],
                                  np.full((nsamples, nlead_steps), np.nan)))
                                 for eval_met in eval_metrics for fcst_prod in fcst_products])

        eval_metric_ds = xr.Dataset(eval_metric_dict, coords={"init_time": np.arange(nsamples),  # just a placeholder
                                                              "fcst_hour": np.arange(nlead_steps)})

        return eval_metric_ds


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
            dates_fcst = pd.to_datetime(coords["fcst_hour"].data)
        except Exception as err:
            print("%{0}: Could not retrieve expected coordinates lat, lon and time_forecast from data.".format(method))
            raise err

        lons, lats = np.meshgrid(lon, lat)

        date0 = dates_fcst[0] - (dates_fcst[1] - dates_fcst[0])
        date0_str = date0.strftime("%Y-%m-%d %H:%M UTC")

        fhhs = ((dates_fcst - date0) / pd.Timedelta('1 hour')).values

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
        for t, fhh in enumerate(fhhs):
            m = Basemap(projection='cyl', llcrnrlat=np.min(lat), urcrnrlat=np.max(lat),
                        llcrnrlon=np.min(lon), urcrnrlon=np.max(lon), resolution='l', ax=axes[t])
            m.drawcoastlines()
            x, y = m(lons, lats)
            if t%6 == 0:
                lat_lab = [1, 0, 0, 0]
                axes[t].set_ylabel(u'Latitude', labelpad=30)
            else:
                lat_lab = list(np.zeros(4))
            if t/6 >= 1:
                lon_lab = [0, 0, 0, 1]
                axes[t].set_xlabel(u'Longitude', labelpad=15)
            else:
                lon_lab = list(np.zeros(4))
            m.drawmapboundary()
            m.drawparallels(np.arange(0, 90, 5),labels=lat_lab, xoffset=1.)
            m.drawmeridians(np.arange(5, 355, 10),labels=lon_lab, yoffset=1.)
            cs = m.contourf(x, y, data.isel(fcst_hour=t)-273.15, clevs, cmap=plt.get_cmap("jet"), ax=axes[t])
            cs_c_pos = m.contour(x, y, data_diff.isel(fcst_hour=t), clevs_diff, linewidths=0.5, ax=axes[t],
                                 colors="black")
            cs_c_neg = m.contour(x, y, data_diff.isel(fcst_hour=t), clevs_diff2, linewidths=1, linestyles="dotted",
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
                                batch_size=args.batch_size, num_stochastic_samples=args.num_stochastic_samples,
                                gpu_mem_frac=args.gpu_mem_frac, seed=args.seed,
                                stochastic_plot_id=args.stochastic_plot_id, args=args)

    test_instance()
    test_instance.run()
    test_instance.handle_eval_metrics()
    test_instance.plot_example_forecasts(metric="mse")


if __name__ == '__main__':
    main()
