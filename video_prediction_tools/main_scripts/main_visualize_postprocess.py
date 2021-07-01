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
from typing import Union, List
# own modules
from normalization import Norm_data
from netcdf_datahandling import get_era5_varatts
from general_utils import check_dir
from metadata import MetaData as MetaData
from main_scripts.main_train_models import *
from data_preprocess.preprocess_data_step2 import *
from model_modules.video_prediction import datasets, models, metrics
from statistical_evaluation import perform_block_bootstrap_metric, avg_metrics, calculate_cond_quantiles, Scores
from postprocess_plotting import plot_avg_eval_metrics, plot_cond_quantile, create_geo_contour_plot


class Postprocess(TrainModel):
    def __init__(self, results_dir: str = None, checkpoint: str= None, mode: str = "test", batch_size: int = None,
                 num_stochastic_samples: int = 1, stochastic_plot_id: int = 0, gpu_mem_frac: float = None,
                 seed: int = None, channel: int = 0, args=None, run_mode: str = "deterministic",
                 eval_metrics: List = ("mse", "psnr", "ssim","acc"), clim_path: str ="/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/T2monthly"):
        """
        Initialization of the class instance for postprocessing (generation of forecasts from trained model +
        basic evauation).
        :param results_dir: output directory to save results
        :param checkpoint: directory point to the model checkpoints
        :param mode: mode of dataset to be processed ("train", "val" or "test"), default: "test"
        :param batch_size: mini-batch size for generating forecasts from trained model
        :param num_stochastic_samples: number of ensemble members for variational models (SAVP, VAE), default: 1
                                       not supported yet!!!
        :param stochastic_plot_id: not supported yet!
        :param gpu_mem_frac: fraction of GPU memory to be pre-allocated
        :param seed: Integer controlling randomization
        :param channel: Channel of interest for statistical evaluation
        :param args: namespace of parsed arguments
        :param run_mode: "deterministic" or "stochastic", default: "deterministic", "stochastic is not supported yet!!!
        :param eval_metrics: metrics used to evaluate the trained model
        :param clim_path:  the path to the climatology nc file
        """
        # copy over attributes from parsed argument
        self.results_dir = self.output_dir = os.path.normpath(results_dir)
        _ = check_dir(self.results_dir, lcreate=True)
        self.batch_size = batch_size
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.set_seed()
        self.num_stochastic_samples = num_stochastic_samples
        #self.num_samples_per_epoch = 20 # reduce number of epoch samples  
        self.stochastic_plot_id = stochastic_plot_id
        self.args = args
        self.checkpoint = checkpoint
        self.clim_path = clim_path
        _ = check_dir(self.checkpoint)
        self.run_mode = run_mode
        self.mode = mode
        self.channel = channel
        # Attributes set during runtime
        self.norm_cls = None
        # configuration of basic evaluation
        self.eval_metrics = eval_metrics
        self.nboots_block = 1000
        self.block_length = 7 * 24  # this corresponds to a block length of 7 days in case of hourly forecasts
        # initialize evrything to get an executable Postprocess instance
        self.save_args_to_option_json()     # create options.json-in results directory
        self.copy_data_model_json()         # copy over JSON-files from model directory
        # get some parameters related to model and dataset
        self.datasplit_dict, self.model_hparams_dict, self.dataset, self.model, self.input_dir_tfr = self.load_jsons()
        self.model_hparams_dict_load = self.get_model_hparams_dict()
        # set input paths and forecast product dictionary
        self.input_dir, self.input_dir_pkl = self.get_input_dirs()
        self.fcst_products = {"persistence": "pfcst", self.model: "mfcst"}
        # correct number of stochastic samples if necessary
        self.check_num_stochastic_samples()
        # get metadata
        md_instance = self.get_metadata()
        self.height, self.width = md_instance.ny, md_instance.nx
        self.vars_in = md_instance.variables
        self.lats, self.lons = md_instance.get_coord_array()
        # get statistics JSON-file
        self.stat_fl = self.set_stat_file()
        self.cond_quantile_vars = self.init_cond_quantile_vars()
        # setup test dataset and model
        self.test_dataset, self.num_samples_per_epoch = self.setup_test_dataset()
        # self.num_samples_per_epoch = 100              # reduced number of epoch samples -> useful for testing
        self.sequence_length, self.context_frames, self.future_length = self.get_data_params()
        self.inputs, self.input_ts = self.make_test_dataset_iterator()
        # set-up model, its graph and do GPU-configuration (from TrainModel)
        self.setup_model()
        self.setup_graph()
        self.setup_gpu_config()
        self.load_climdata()
    # Methods that are called during initialization
    def get_input_dirs(self):
        """
        Retrieves top-level input directory and nested pickle-directory from input_dir_tfr
        :return input_dir: top-level input-directoy
        :return input_dir_pkl: Input directory where pickle-files are placed
        """
        method = Postprocess.get_input_dirs.__name__

        if not hasattr(self, "input_dir_tfr"):
            raise AttributeError("Attribute input_dir_tfr is still missing.".format(method))

        _ = check_dir(self.input_dir_tfr)

        input_dir = os.path.dirname(self.input_dir_tfr.rstrip("/"))
        input_dir_pkl = os.path.join(input_dir, "pickle")

        _ = check_dir(input_dir_pkl)

        return input_dir, input_dir_pkl

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
        :return datasplit_dict: path to datasplit-dictionary JSON-file of trained model
        :return model_hparams_dict: path to model hyperparameter-dictionary JSON-file of trained model
        :return dataset: Name of datset used to train model
        :return model: Name of trained model
        :return input_dir_tfr: path to input directory where TF-records are stored
        """
        method_name = Postprocess.load_jsons.__name__

        datasplit_dict = os.path.join(self.results_dir, "data_dict.json")
        model_hparams_dict = os.path.join(self.results_dir, "model_hparams.json")
        checkpoint_opt_dict = os.path.join(self.results_dir, "options_checkpoints.json")

        # sanity checks on the JSON-files
        if not os.path.isfile(datasplit_dict):
            raise FileNotFoundError("%{0}: The file data_dict.json is missing in {1}".format(method_name,
                                                                                             self.results_dir))

        if not os.path.isfile(model_hparams_dict):
            raise FileNotFoundError("%{0}: The file model_hparams.json is missing in {1}".format(method_name,
                                                                                                 self.results_dir))

        if not os.path.isfile(checkpoint_opt_dict):
            raise FileNotFoundError("%{0}: The file options_checkpoints.json is missing in {1}"
                                    .format(method_name, self.results_dir))
        # retrieve some data from options_checkpoints.json
        try:
            with open(checkpoint_opt_dict) as f:
                options_checkpoint = json.loads(f.read())
                dataset = options_checkpoint["dataset"]
                model = options_checkpoint["model"]
                input_dir_tfr = options_checkpoint["input_dir"]
        except Exception as err:
            print("%{0}: Something went wrong when reading the checkpoint-file '{1}'".format(method_name,
                                                                                             checkpoint_opt_dict))
            raise err

        return datasplit_dict, model_hparams_dict, dataset, model, input_dir_tfr

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
        #print('self.lats: ',self.lats)
        return md_instance

    def load_climdata(self,clim_path="/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/T2monthly",
                            var="T2M",climatology_fl="climatology_t2m_1991-2020.nc"):
        """
        params:climatology_fl: str, the full path to the climatology file
        params:var           : str, the variable name 
        
        """
        data_clim_path = os.path.join(clim_path,climatology_fl)
        data = xr.open_dataset(data_clim_path)
        dt_clim = data[var]

        clim_lon = dt_clim['lon'].data
        clim_lat = dt_clim['lat'].data
        
        meta_lon_loc = np.zeros((len(clim_lon)), dtype=bool)
        for i in range(len(clim_lon)):
            if np.round(clim_lon[i],1) in self.lons.data:
                meta_lon_loc[i] = True

        meta_lat_loc = np.zeros((len(clim_lat)), dtype=bool)
        for i in range(len(clim_lat)):
            if np.round(clim_lat[i],1) in self.lats.data:
                meta_lat_loc[i] = True

        # get the coordinates of the data after running CDO
        coords = dt_clim.coords
        nlat, nlon = len(coords["lat"]), len(coords["lon"])
        # modify it our needs
        coords_new = dict(coords)
        coords_new.pop("time")
        coords_new["month"] = np.arange(1, 13) 
        coords_new["hour"] = np.arange(0, 24)
        # initialize a new data array with explicit dimensions for month and hour
        data_clim_new = xr.DataArray(np.full((12, 24, nlat, nlon), np.nan), coords=coords_new, dims=["month", "hour", "lat", "lon"])
        # do the reorganization
        for month in np.arange(1, 13): 
            data_clim_new.loc[dict(month=month)]=dt_clim.sel(time=dt_clim["time.month"]==month)

        self.data_clim = data_clim_new[dict(lon=meta_lon_loc,lat=meta_lat_loc)]
        print("self.data_clim",self.data_clim) 
         
    def setup_test_dataset(self):
        """
        setup the test dataset instance
        :return test_dataset: the test dataset instance
        """
        VideoDataset = datasets.get_dataset_class(self.dataset)
        test_dataset = VideoDataset(input_dir=self.input_dir_tfr, mode=self.mode, datasplit_config=self.datasplit_dict)
        nsamples = test_dataset.num_examples_per_epoch()

        return test_dataset, nsamples

    def get_data_params(self):
        """
        Get the context_frames, future_frames and total frames from hparamters settings.
        Note that future_frames_length is the number of predicted frames.
        """
        method = Postprocess.get_data_params.__name__

        if not hasattr(self, "model_hparams_dict_load"):
            raise AttributeError("%{0}: Attribute model_hparams_dict_load is still unset.".format(method))

        try:
            context_frames = self.model_hparams_dict_load["context_frames"]
            sequence_length = self.model_hparams_dict_load["sequence_length"]
        except Exception as err:
            print("%{0}: Could not retrieve context_frames and sequence_length from model_hparams_dict_load-attribute"
                  .format(method))
            raise err
        future_length = sequence_length - context_frames
        if future_length <= 0:
            raise ValueError("Calculated future_length must be greater than zero.".format(method))

        return sequence_length, context_frames, future_length

    def set_stat_file(self):
        """
        Set the name of the statistic file from the input directory
        :return stat_fl: Path to statistics JSON-file of input data used to train the model
        """
        method = Postprocess.set_stat_file.__name__

        if not hasattr(self, "input_dir"):
            raise AttributeError("%{0}: Attribute input_dir is still unset".format(method))

        stat_fl = os.path.join(self.input_dir, "statistics.json")
        if not os.path.isfile(stat_fl):
            raise FileNotFoundError("%{0}: Cannot find statistics JSON-file '{1}'".format(method, stat_fl))

        return stat_fl

    def init_cond_quantile_vars(self):
        """
        Get a list of variable names for conditional quantile plot
        :return cond_quantile_vars: list holding the variable names of interest
        """
        method = Postprocess.init_cond_quantile_vars.__name__

        if not hasattr(self, "model"):
            raise AttributeError("%{0}: Attribute model is still unset.".format(method))
        cond_quantile_vars = ["{0}_{1}_fcst".format(self.vars_in[self.channel], self.model),
                              "{0}_ref".format(self.vars_in[self.channel])]

        return cond_quantile_vars

    def make_test_dataset_iterator(self):
        """
        Make the dataset iterator
        """
        method = Postprocess.make_test_dataset_iterator.__name__

        if not hasattr(self, "test_dataset"):
            raise AttributeError("%{0}: Attribute test_dataset is still unset".format(method))

        if not hasattr(self, "batch_size"):
            raise AttributeError("%{0}: Attribute batch_sie is still unset".format(method))

        test_tf_dataset = self.test_dataset.make_dataset(self.batch_size)
        test_iterator = test_tf_dataset.make_one_shot_iterator()
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        test_handle = test_iterator.string_handle()
        dataset_iterator = tf.data.Iterator.from_string_handle(test_handle, test_tf_dataset.output_types,
                                                               test_tf_dataset.output_shapes)
        input_iter = dataset_iterator.get_next()
        ts_iter = input_iter["T_start"]

        return input_iter, ts_iter

    def check_num_stochastic_samples(self):
        """
        stochastic forecasting only suitable for the geneerate models such as SAVP, vae.
        For convLSTM, McNet only do determinstic forecasting
        """
        method = Postprocess.check_num_stochastic_samples.__name__

        if not hasattr(self, "model"):
            raise AttributeError("%{0}: Attribute model is still unset".format(method))
        if not hasattr(self, "num_stochastic_samples"):
            raise AttributeError("%{0}: Attribute num_stochastic_samples is still unset".format(method))

        if self.model == "convLSTM" or self.model == "test_model" or self.model == 'mcnet':
            if self.num_stochastic_samples > 1:
                print("Number of samples for deterministic model cannot be larger than 1. Higher values are ignored.")
            self.num_stochastic_samples = 1

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
        # init sample index for looping
        sample_ind = 0
        nsamples = self.num_samples_per_epoch
        # initialize xarray datasets
        eval_metric_ds = Postprocess.init_metric_ds(self.fcst_products, self.eval_metrics, self.vars_in[self.channel],
                                                    nsamples, self.future_length)
        cond_quantiple_ds = None

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
            # store data into datset & get number of samples (may differ from batch_size at the end of the test dataset)
            times_0, init_times = self.get_init_time(t_starts)
            batch_ds = self.create_dataset(input_images_denorm, gen_images_denorm, init_times)
            nbs = np.minimum(self.batch_size, self.num_samples_per_epoch - sample_ind)
            batch_ds = batch_ds.isel(init_time=slice(0, nbs))

            for i in np.arange(nbs):
                # work-around to make use of get_persistence_forecast_per_sample-method
                times_seq = (pd.date_range(times_0[i], periods=int(self.sequence_length), freq="h")).to_pydatetime()
                print('times_seq: ',times_seq)
                # get persistence forecast for sequences at hand and write to dataset
                persistence_seq, _ = Postprocess.get_persistence(times_seq, self.input_dir_pkl)
                for ivar, var in enumerate(self.vars_in):
                    batch_ds["{0}_persistence_fcst".format(var)].loc[dict(init_time=init_times[i])] = \
                        persistence_seq[self.context_frames-1:, :, :, ivar]

                # save sequences to netcdf-file and track initial time
                nc_fname = os.path.join(self.results_dir, "vfp_date_{0}_sample_ind_{1:d}.nc"
                                        .format(pd.to_datetime(init_times[i]).strftime("%Y%m%d%H"), sample_ind + i))
                
                if os.path.exists(nc_fname):
                    print("The file {} exist".format(nc_fname))
                else:
                    self.save_ds_to_netcdf(batch_ds.isel(init_time=i), nc_fname)

                # end of batch-loop
            # write evaluation metric to corresponding dataset and sa
            eval_metric_ds = self.populate_eval_metric_ds(eval_metric_ds, batch_ds, sample_ind,
                                                          self.vars_in[self.channel])
            cond_quantiple_ds = Postprocess.append_ds(batch_ds, cond_quantiple_ds, self.cond_quantile_vars, "init_time", dtype=np.float16)
            # ... and increment sample_ind
            sample_ind += self.batch_size
            # end of while-loop for samples
        # safe dataset with evaluation metrics for later use
        self.eval_metrics_ds = eval_metric_ds
        self.cond_quantiple_ds = cond_quantiple_ds
        #self.add_ensemble_dim()

    # all methods of the run factory
    def init_session(self):
        """
        Initialize TensorFlow-session
        :return: -
        """
        method = Postprocess.init_session.__name__

        if not hasattr(self, "config"):
            raise AttributeError("Attribute config is still unset.".format(method))

        self.sess = tf.Session(config=self.config)
        self.sess.graph.as_default()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

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
        dims = ["lat", "lon"]
        eval_metrics_func = [Scores(metric,dims).score_func for metric in self.eval_metrics]
        varname_ref = "{0}_ref".format(varname)
        # reset init-time coordinate of metric_ds in place and get indices for slicing
        ind_end = np.minimum(ind_start + self.batch_size, self.num_samples_per_epoch)
        init_times_metric = metric_ds["init_time"].values
        init_times_metric[ind_start:ind_end] = data_ds["init_time"]
        metric_ds = metric_ds.assign_coords(init_time=init_times_metric)
        print("metric_ds",metric_ds)
        # populate metric_ds
        for fcst_prod in self.fcst_products.keys():
            for imetric, eval_metric in enumerate(self.eval_metrics):
                metric_name = "{0}_{1}_{2}".format(varname, fcst_prod, eval_metric)
                varname_fcst = "{0}_{1}_fcst".format(varname, fcst_prod)
                dict_ind = dict(init_time=data_ds["init_time"])
                print('metric_name: ',metric_name)
                print('varname_fcst: ',varname_fcst)
                print('varname_ref: ',varname_ref)
                print('dict_ind: ',dict_ind)
                print('fcst_prod: ',fcst_prod)
                print('imetric: ',imetric)
                print('eval_metric: ',eval_metric)
                metric_ds[metric_name].loc[dict_ind] = eval_metrics_func[imetric](data_fcst=data_ds[varname_fcst],
                                                                                  data_ref=data_ds[varname_ref],
                                                                                  data_clim=self.data_clim)
                print('data_ds[varname_fcst] shape: ',data_ds[varname_fcst].shape)
                print('metric_ds[metric_name].loc[dict_ind] shape: ',metric_ds[metric_name].loc[dict_ind].shape)
                print('metric_ds[metric_name].loc[dict_ind]: ',metric_ds[metric_name].loc[dict_ind])
            # end of metric-loop
        # end of forecast product-loop
        
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
                                                                   varname=ivar)
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
        _ = plot_avg_eval_metrics(self.eval_metrics_ds, self.eval_metrics, self.fcst_products,
                                  self.vars_in[self.channel], self.results_dir)

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
                                                      int(quantiles[i] * 100.)))

                create_geo_contour_plot(data_fcst, data_diff, varname, plt_fname_base)

    def plot_conditional_quantiles(self):

        # release some memory
        Postprocess.clean_obj_attribute(self, "eval_metrics_ds")

        # the variables for conditional quantile plot
        var_fcst = self.cond_quantile_vars[0]
        var_ref = self.cond_quantile_vars[1]

        data_fcst = get_era5_varatts(self.cond_quantiple_ds[var_fcst], self.cond_quantiple_ds[var_fcst].name)
        data_ref = get_era5_varatts(self.cond_quantiple_ds[var_ref], self.cond_quantiple_ds[var_ref].name)

        # create plots
        fhhs = data_fcst.coords["fcst_hour"]
        for hh in fhhs:
            # calibration refinement factorization
            plt_fname_cf = os.path.join(self.results_dir, "cond_quantile_{0}_{1}_fh{2:0d}_calibration_refinement.png"
                                        .format(self.vars_in[self.channel], self.model, int(hh)))

            quantile_panel_cf, cond_variable_cf = calculate_cond_quantiles(data_fcst.sel(fcst_hour=hh),
                                                                           data_ref.sel(fcst_hour=hh),
                                                                           factorization="calibration_refinement",
                                                                           quantiles=(0.05, 0.5, 0.95))

            plot_cond_quantile(quantile_panel_cf, cond_variable_cf, plt_fname_cf)

            # likelihood-base rate factorization
            plt_fname_lbr = plt_fname_cf.replace("calibration_refinement", "likelihood-base_rate")
            quantile_panel_lbr, cond_variable_lbr = calculate_cond_quantiles(data_fcst.sel(fcst_hour=hh),
                                                                             data_ref.sel(fcst_hour=hh),
                                                                             factorization="likelihood-base_rate",
                                                                             quantiles=(0.05, 0.5, 0.95))

            plot_cond_quantile(quantile_panel_lbr, cond_variable_lbr, plt_fname_lbr)

    @staticmethod
    def clean_obj_attribute(obj, attr_name, lremove=False):
        """
        Cleans attribute of object by setting it to None (can be used to releave memory)
        :param obj: the object/ class instance
        :param attr_name: the attribute from the object to be cleaned
        :param lremove: flag if attribute is removed or set to None
        :return: the object/class instance with the attribute's value changed to None
        """
        method = Postprocess.clean_obj_attribute.__name__

        if not hasattr(obj, attr_name):
            print("%{0}: Class attribute '{1}' does not exist. Nothing to do...".format(method, attr_name))
        else:
            if lremove:
                delattr(obj, attr_name)
            else:
                setattr(obj, attr_name, None)

        return obj

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
            # print("time_pickle_second:", time_pickle_second)
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
    def append_ds(ds_in: xr.Dataset, ds_preexist: xr.Dataset, varnames: list, dim2append: str, dtype=None):
        """
        Append existing datset with subset of dataset based on selected variables
        :param ds_in: the input dataset from which variables should be retrieved
        :param ds_preexist: the accumulator datsaet to be appended (can be initialized with None)
        :param dim2append:
        :param varnames: List of variables that should be retrieved from ds_in and that are appended to ds_preexist
        :return: appended version of ds_preexist
        """
        method = Postprocess.append_ds.__name__

        varnames_str = ",".join(varnames)
        # sanity checks
        if not isinstance(ds_in, xr.Dataset):
            raise ValueError("%{0}: ds_in must be a xarray dataset, but is of type {1}".format(method, type(ds_in)))

        if not set(varnames).issubset(ds_in.data_vars):
            raise ValueError("%{0}: Could not find all variables ({1}) in input dataset ds_in.".format(method,
                                                                                                       varnames_str))
        #Bing : why using dtype as an aurument since it seems you only want ton configure dtype as np.double
        if dtype is None:
            dtype = np.double
        else:
            if not isinstance(dtype, type(np.double)):
                raise ValueError("%{0}: dytpe must be a NumPy datatype, but is of type '{1}'".format(method, type(dtype)))
  
        if ds_preexist is None:
            ds_preexist = ds_in[varnames].copy(deep=True)
            ds_preexist = ds_preexist.astype(dtype)                           # change data type (if necessary)
            return ds_preexist
        else:
            if not isinstance(ds_preexist, xr.Dataset):
                raise ValueError("%{0}: ds_preexist must be a xarray dataset, but is of type {1}"
                                 .format(method, type(ds_preexist)))
            if not set(varnames).issubset(ds_preexist.data_vars):
                raise ValueError("%{0}: Could not find all varibales ({1}) in pre-existing dataset ds_preexist"
                                 .format(method, varnames_str))

        try:
            ds_preexist = xr.concat([ds_preexist, ds_in[varnames].astype(dtype)], dim2append)
        except Exception as err:
            print("%{0}: Failed to concat datsets along dimension {1}.".format(method, dim2append))
            print(ds_in)
            print(ds_preexist)
            raise err

        return ds_preexist

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
        eval_metric_dict = dict([("{0}_{1}_{2}".format(varname, *(fcst_prod, eval_met)), (["init_time", "fcst_hour"],
                                  np.full((nsamples, nlead_steps), np.nan)))
                                 for eval_met in eval_metrics for fcst_prod in fcst_products])

        init_time_dummy = pd.date_range("1900-01-01 00:00", freq="s", periods=nsamples)
        eval_metric_ds = xr.Dataset(eval_metric_dict, coords={"init_time": init_time_dummy,  # just a placeholder
                                                              "fcst_hour": np.arange(1, nlead_steps+1)})

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
    parser.add_argument("--gpu_mem_frac", type=float, default=0.95, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluation_metrics", "-eval_metrics", dest="eval_metrics", nargs="+", default=("mse", "psnr", "ssim","acc"),
                        help="Metrics to be evaluate the trained model. Must be known metrics, see Scores-class.")
    parser.add_argument("--channel", "-channel", dest="channel", type=int, default=0,
                        help="Channel which is used for evaluation.")
    args = parser.parse_args()

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    # initialize postprocessing instance
    postproc_instance = Postprocess(results_dir=args.results_dir, checkpoint=args.checkpoint, mode="test",
                                    batch_size=args.batch_size, num_stochastic_samples=args.num_stochastic_samples,
                                    gpu_mem_frac=args.gpu_mem_frac, seed=args.seed, args=args,
                                    eval_metrics=args.eval_metrics, channel=args.channel)
    # run the postprocessing
    postproc_instance.run()
    postproc_instance.handle_eval_metrics()
    postproc_instance.plot_example_forecasts(metric=args.eval_metrics[0], channel=args.channel)
    postproc_instance.plot_conditional_quantiles()


if __name__ == '__main__':
    main()
