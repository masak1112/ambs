# coding=utf-8
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
# SPDX-FileCopyrightText: 2018 Alex X. Lee
#
# SPDX-License-Identifier: MIT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Michael Langguth"
__date__ = "2020-10-22"

import os, glob
import argparse
import errno
import json
from typing import Union, List
import random
import time
import numpy as np
import xarray as xr
import tensorflow as tf
from model_modules.video_prediction import models
from model_modules.video_prediction.datasets import get_dataset
import matplotlib.pyplot as plt
import pickle as pkl
from model_modules.video_prediction.utils import tf_utils
from general_utils import *
import math
import shutil
from pathlib import Path

class TrainModel(object):
    def __init__(self, input_dir: str = None, output_dir: str = None, datasplit_dict: str = None,
                 model_hparams_dict: str = None, model: str = None, checkpoint: str = None, dataset_name: str = None,
                 gpu_mem_frac: float = 1., seed: int = None, args=None, diag_intv_frac: float = 0.001,
                 frac_start_save: float = None, frac_intv_save: float = None):
        """
        Class instance for training the models
        :param input_dir: parent directory under which "pickle" and "tfrecords" files directiory are located
        :param output_dir: directory where all the output is saved (e.g. model, JSON-files, training curves etc.)
        :param datasplit_dict: JSON-file for defining data splitting
        :param model_hparams_dict: JSON-file of model hyperparameters
        :param model: model class name
        :param checkpoint: checkpoint directory (pre-trained models)
        :param dataset: dataset class name
        :param gpu_mem_frac: fraction of GPU memory to be preallocated
        :param seed: seed of the randomizers
        :param args: list of arguments passed
        :param diag_intv_frac: interval for diagnozing the model (create loss-curves and save pickle-file with losses)
        :param frac_start_save: fraction of total iterations steps to start checkpointing the model
        :param frac_intv_save: fraction of total iterations steps for checkpointing the model
        """
        self.input_dir = Path(input_dir).resolve(strict=False)
        self.output_dir = Path(output_dir).resolve(strict=False)
        self.datasplit_dict = datasplit_dict
        self.model_hparams_dict = model_hparams_dict
        self.checkpoint = checkpoint
        self.dataset_name = dataset_name
        self.model = model
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.args = args
        self.diag_intv_frac = diag_intv_frac
        self.frac_start_save = frac_start_save
        self.frac_intv_save = frac_intv_save
        # for diagnozing and saving the model during training
        self.saver_loss = None         # set in create_fetches_for_train-method
        self.saver_loss_name = None    # set in create_fetches_for_train-method 
        self.saver_loss_dict = None    # set in create_fetches_for_train-method if loss of interest is nested 
        self.diag_intv_step = None     # set in calculate_samples_and_epochs-method

    def setup(self):
        self.set_seed()
        self.get_model_hparams_dict()
        self.load_params_from_checkpoints_dir()
        self.setup_datasets()
        self.make_dataset_iterator()
        self.setup_model()
        self.setup_graph()
        self.save_dataset_model_params_to_checkpoint_dir(dataset=self.dataset, video_model=self.video_model) # TODO: resolve potetial incompatibility
        self.count_parameters()
        self.create_saver_and_writer()
        self.setup_gpu_config()
        self.calculate_checkpoint_saver_conf()

    def set_seed(self):
        """
        Set seed to control the same train/val/testing dataset for the same seed
        """
        if self.seed is not None:
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    def check_output_dir(self):
        """
        Checks if output directory is existing.
        """
        method = TrainModel.check_output_dir.__name__

        if self.output_dir is None:
            raise ValueError("%{0}: Output_dir-argument is empty. Please define a proper output_dir".format(method))
        elif not os.path.isdir(self.output_dir):
            raise NotADirectoryError("Base output_dir {0} does not exist. Pass a proper output_dir".format(method) +
                                     " and make use of env_setup/generate_runscript.py.")

    def get_model_hparams_dict(self):
        """
        Get and read model_hparams_dict from json file to dictionary 
        """
        if self.model_hparams_dict:
            with open(self.model_hparams_dict, 'r') as f:
                print("self.model_hparams_dict",self.model_hparams_dict)
                self.model_hparams_dict_load = json.loads(f.read())
        else:
            raise FileNotFoundError("hparam directory doesn't exist! please check {}!".format(self.model_hparams_dict))

        return self.model_hparams_dict_load

    def load_params_from_checkpoints_dir(self):
        """
        If checkpoint is none, load and read the json files of datasplit_config, and hparam_config,
        and use the corresponding parameters.
        If the checkpoint is given, the configuration of dataset, model and options in the checkpoint dir will be
        restored and used for continue training.
        """
        method = TrainModel.load_params_from_checkpoints_dir.__name__

        if self.checkpoint:
            self.checkpoint_dir = os.path.normpath(self.checkpoint)
            if not os.path.isdir(self.checkpoint):
                self.checkpoint_dir, _ = os.path.split(self.checkpoint_dir)
            if not os.path.exists(self.checkpoint_dir):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.checkpoint_dir)
            # read and overwrite dataset and model from checkpoint
            try:
                with open(os.path.join(self.checkpoint_dir, "options.json")) as f:
                    print("%{0}: Loading options from checkpoint '{1}'".format(method, self.checkpoint))
                    self.options = json.loads(f.read())
                    self.dataset_name = self.dataset_name or self.options['dataset']
                    self.model = self.model or self.options['model']
            except FileNotFoundError:
                print("%{0}: options.json does not exist in {1}".format(method, self.checkpoint_dir))
            # loading hyperparameters from checkpoint
            try:
                with open(os.path.join(self.checkpoint_dir, "model_hparams.json")) as f:
                    self.model_hparams_dict_load.update(json.loads(f.read()))
            except FileNotFoundError:
                print("%{0}: model_hparams.json does not exist in {1}".format(method, self.checkpoint_dir))
                
    def setup_datasets(self):
        """
        Setup train and val dataset instance with the corresponding data split configuration.
        Simultaneously, sequence_length is attached to the hyperparameter dictionary.
        """
        # get some parameters from the model hyperparameters
        self.batch_size = self.model_hparams_dict_load["batch_size"]
        self.max_epochs = self.model_hparams_dict_load["max_epochs"]
        # create dataset instance

        self.dataset = get_dataset(self.dataset_name, input_dir=self.input_dir, output_dir=self.output_dir, datasplit_path=self.datasplit_dict, hparams_path=self.model_hparams_dict, seed=self.seed)
        
        self.calculate_samples_and_epochs()
        self.model_hparams_dict_load.update({"sequence_length": self.dataset.sequence_length})

    def setup_model(self, mode="train"):
        """
        Set up model instance for the given model names
        :param mode: "train" used the model graph in train process;  "test" for postprocessing step
        """
        VideoPredictionModel = models.get_model_class(self.model)
        self.video_model = VideoPredictionModel(hparams_dict_config=self.model_hparams_dict, mode=mode)

    def setup_graph(self):
        """
        build model graph
        """
        self.video_model.build_graph(self.inputs)
        
    def make_dataset_iterator(self):
        """
        Prepare the dataset interator for training and validation
        """
        self.batch_size = self.model_hparams_dict_load["batch_size"]
        train_tf_dataset = self.dataset.make_training()
        train_iterator = train_tf_dataset.make_one_shot_iterator()
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.train_handle = train_iterator.string_handle()
        val_tf_dataset = self.dataset.make_validation()
        val_iterator = val_tf_dataset.make_one_shot_iterator()
        self.val_handle = val_iterator.string_handle()
        self.iterator = tf.data.Iterator.from_string_handle(
            self.train_handle, train_tf_dataset.output_types, train_tf_dataset.output_shapes)
        self.inputs = self.iterator.get_next()
        # since era5 tfrecords include T_start, we need to remove it from the tfrecord when we train SAVP
        # Otherwise an error will be risen by SAVP 
        if self.dataset_name == "era5" and self.model == "savp":
            del self.inputs["T_start"]

    def save_dataset_model_params_to_checkpoint_dir(self, dataset, video_model):
        """
        Save all setup configurations such as args, data_hparams, and model_hparams into output directory
        """
        with open(os.path.join(self.output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(self.args), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "dataset_hparams.json"), "w") as f:
            f.write(json.dumps(dataset.hparams, sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "model_hparams.json"), "w") as f:
            print("video_model.get_hparams",video_model.get_hparams)
            f.write(json.dumps(video_model.get_hparams, sort_keys=True, indent=4))
        #with open(os.path.join(self.output_dir, "data_dict.json"), "w") as f:
        #   f.write(json.dumps(dataset.data_dict, sort_keys=True, indent=4))

    def count_parameters(self):
        """
        Count the paramteres of the model
        """ 
        with tf.name_scope("parameter_count"):
            # exclude trainable variables that are replicates (used in multi-gpu setting)
            self.trainable_variables = set(tf.trainable_variables()) & set(self.video_model.saveable_variables)
            self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in self.trainable_variables])

    def create_saver_and_writer(self):
        """
        Create saver to save the models latest checkpoints, and a summery writer to store the train/val metrics  
        """
        self.saver = tf.train.Saver(var_list=self.video_model.saveable_variables, max_to_keep=None)
        self.summary_writer = tf.summary.FileWriter(self.output_dir)

    def setup_gpu_config(self):
        """
        Setup GPU options 
        """
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_mem_frac, allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=self.gpu_options, allow_soft_placement=True)

    def calculate_samples_and_epochs(self):
        """
        Calculate the number of samples for train dataset, which is used for each epoch training
        Calculate the iterations (samples multiple by max_epochs) for training.
        """
        method = TrainModel.calculate_samples_and_epochs.__name__        

        self.num_examples = self.dataset.num_training_samples
        self.steps_per_epoch = int(self.num_examples/self.batch_size)
        self.total_steps = self.steps_per_epoch * self.max_epochs
        self.diag_intv_step = int(self.diag_intv_frac*self.total_steps)

        if self.diag_intv_step == 0:
            self.diag_intv_step = 1
        else:
            pass
        print("%{}: Batch size: {}; max_epochs: {}; num_samples per epoch: {}; steps_per_epoch: {}, total steps: {}"
              .format(method, self.batch_size, self.max_epochs, self.num_examples, self.steps_per_epoch,
                      self.total_steps))



    def calculate_checkpoint_saver_conf(self):
        """
        Calculate the start step for saving the checkpoint, and the frequences steps to save model

        """
        method = TrainModel.calculate_checkpoint_saver_conf.__name__

        if not hasattr(self, "total_steps"):
            raise RuntimeError("%{0} self.total_steps is still unset. Run calculate_samples_and_epochs beforehand"
                               .format(method))
        if self.frac_intv_save > 1 or self.frac_intv_save<0 :
            raise ValueError("%{0}: frac_intv_save must be less than 1 and larger than 0".format(method))
        if self.frac_start_save > 1 or self.frac_start_save < 0:
            raise ValueError("%{0}: frac_start_save must be less than 1 and larger than 0".format(method))

        self.chp_start_step = int(math.ceil(self.total_steps * self.frac_start_save))
        self.chp_intv_step = int(math.ceil(self.total_steps * self.frac_intv_save))
        print("%{0}: Model will be saved after step {1:d} at each {2:d} interval step "
              .format(method, self.chp_start_step,self.chp_intv_step))

    def restore(self, sess, checkpoints, restore_to_checkpoint_mapping=None):
        """
        Restore the models checkpoints if the checkpoints is given
        """
        method = TrainModel.restore.__name__

        if checkpoints is None:
            print("%{0}: Checkpoint is None!".format(method))
        elif os.path.isdir(checkpoints) and (not os.path.exists(os.path.join(checkpoints, "checkpoint"))):
            print("%{0}: There are no checkpoints in the dir {1}".format(method, checkpoints))
        else:
            var_list = self.video_model.saveable_variables
            # possibly restore from multiple checkpoints. useful if subset of weights
            # (e.g. generator or discriminator) are on different checkpoints.
            if not isinstance(checkpoints, (list, tuple)):
                checkpoints = [checkpoints]
            # automatically skip global_step if more than one checkpoint is provided
            skip_global_step = len(checkpoints) > 1
            savers = []
            for checkpoint in checkpoints:
                print("%{0}: creating restore saver from checkpoint {1}".format(method, checkpoint))
                saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, var_list,
                                                                 skip_global_step=skip_global_step,
                                                                 restore_to_checkpoint_mapping=restore_to_checkpoint_mapping)
                savers.append(saver)
            restore_op = [saver.saver_def.restore_op_name for saver in savers]
            sess.run(restore_op)
    
    def restore_train_val_losses(self):
        """
        Restore the train and validation losses in the pickle file if checkpoint is given 
        """
        if self.checkpoint is None:
            train_losses, val_losses = [], []
        elif os.path.isdir(self.checkpoint) and (not os.path.exists(os.path.join(self.output_dir, "checkpoint"))):
            train_losses,val_losses = [], []
        else:
            with open(os.path.join(self.output_dir, "train_losses.pkl"), "rb") as f:
                train_losses = pkl.load(f)
            with open(os.path.join(self.output_dir, "val_losses.pkl"), "rb") as f:
                val_losses = pkl.load(f)
        return train_losses,val_losses

    def create_checkpoints_folder(self, step:int=None):
        """
        Create a folder to store checkpoint at certain step.
        :param step: the iteration step corresponding to the checkpoint
        return : dir path to save model
        """
        full_dir_name = os.path.join(self.output_dir, "checkpoint_{0:d}".format(step))
        os.makedirs(full_dir_name, exist_ok=True)
        return full_dir_name

    def train_model(self):
        """
        Start session and train the model by looping over all iteration steps
        """
        method = TrainModel.train_model.__name__

        self.global_step = tf.train.get_or_create_global_step()
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.restore(sess, self.checkpoint)
            start_step = sess.run(self.global_step)
            print("%{0}: Iteration starts at step {1}".format(method, start_step))
            # start at one step earlier to log everything without doing any training
            # step is relative to the start_step
            train_losses, val_losses = self.restore_train_val_losses()
            # initialize auxiliary variables
            time_per_iteration = []
            run_start_time = time.time()
            # perform iteration
            for step in range(start_step, self.total_steps):
                timeit_start = time.time()
                # Run training data
                self.create_fetches_for_train()             # In addition to the loss, we fetch the optimizer
                self.results = sess.run(self.fetches)       # ...and run it here!
                # Note: For SAVP, the obtained loss is a list where the first element is of interest, for convLSTM,
                # it's just a number. Thus, with ensure_list(<losses>)[0], we can handle both
                train_losses.append(ensure_list(self.results[self.saver_loss])[0])
                # run and fetch losses for validation data
                val_handle_eval = sess.run(self.val_handle)
                self.create_fetches_for_val()
                self.val_results = sess.run(self.val_fetches, feed_dict={self.train_handle: val_handle_eval})
                val_losses.append(ensure_list(self.val_results[self.saver_loss])[0])
                self.write_to_summary()
                self.print_results(step, self.results)
                # track iteration time
                time_iter = time.time() - timeit_start
                time_per_iteration.append(time_iter)
                print("%{0}: time needed for this step {1:.3f}s".format(method, time_iter))
                if (step >= self.chp_start_step and (step-self.chp_start_step)%self.chp_intv_step == 0) or \
                    step == self.total_steps - 1:
                    #create a checkpoint folder for step
                    full_dir_name = self.create_checkpoints_folder(step=step)
                    self.saver.save(sess, os.path.join(full_dir_name, "model_"), global_step=step)

                # pickle file and plots are always created
                if step % self.diag_intv_step == 0 or step == self.total_steps - 1:
                    TrainModel.save_results_to_pkl(train_losses, val_losses, self.output_dir)
                    TrainModel.plot_train(train_losses, val_losses, self.saver_loss_name, self.output_dir)

            # Final diagnostics: training track time and save to pickle-files)
            train_time = time.time() - run_start_time

            avg_time_first_epoch = np.mean(time_per_iteration[:self.steps_per_epoch])
            avg_time_non_first_epoch = np.mean(time_per_iteration[self.steps_per_epoch:])
            results_dict = {"train_time": train_time, "total_steps": self.total_steps,
                            "avg_time_first_epoch": avg_time_first_epoch,
                            "avg_time_non_first_epoch":avg_time_non_first_epoch}

            TrainModel.save_results_to_dict(results_dict, self.output_dir)

            print("%{0}: Training loss decreased from {1:.6f} to {2:.6f}:"
                  .format(method, np.mean(train_losses[0:10]), np.mean(train_losses[-self.diag_intv_step:])))
            print("%{0}: Validation loss decreased from {1:.6f} to {2:.6f}:"
                  .format(method, np.mean(val_losses[0:10]), np.mean(val_losses[-self.diag_intv_step:])))
            print("%{0}: Training finished".format(method))
            print("%{0}: Total training time: {1:.2f} min".format(method, train_time/60.))
            print("%{0}: The average of training time for the first epoch: {1:.2f} sec".format(method, avg_time_first_epoch))
            print("%{0}: The average of training time for after first epoch: {1:.2f} sec".format(method,avg_time_non_first_epoch))
            return train_time, time_per_iteration
 
    def create_fetches_for_train(self):
        """
        Fetch variables in the graph, this can be custermized based on models and also the needs of users
        """
        # This is the basic fetch for all the models
        fetch_list = ["train_op", "summary_op", "global_step","total_loss"]

        # Append fetches depending on model to be trained
        if self.video_model.__class__.__name__ == "ConvLstmGANVideoPredictionModel":
            self.saver_loss = fetch_list[-1]  
            self.saver_loss_name = "Loss"
        if self.video_model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel":
            fetch_list = fetch_list + ["inputs"]
            self.saver_loss = fetch_list[-1]
        else: 
            self.saver_loss = "total_loss"
        self.fetches = self.generate_fetches(fetch_list)
        return self.fetches

    def create_fetches_for_val(self):
        """
        Fetch variables in the graph for validation dataset, customized depending on models and users' needs
        """
        method = TrainModel.create_fetches_for_val.__name__

        if not self.saver_loss:
            raise AttributeError("%{0}: saver_loss is still not set. create_fetches_for_train must be run in advance."
                                 .format(method))
        if self.saver_loss_dict:
            fetch_list = ["summary_op", (self.saver_loss_dict, self.saver_loss)]
        else:
            fetch_list= ["summary_op", self.saver_loss]

        self.val_fetches = self.generate_fetches(fetch_list)

        return self.val_fetches

    def generate_fetches(self, fetch_list):
        """
        Generates dictionary of fetches from video model instance
        :param fetch_list: list of attributes of video model instance that are of particular interest; 
                           can also handle tuples as list-elements to get attributes nested in a dictionary
        :return: dictionary of fetches with keys from fetch_list and values from video model instance
        """
        method = TrainModel.generate_fetches.__name__

        if not self.video_model:
            raise AttributeError("%{0}: video_model is still not set. setup_model must be run in advance."
                                 .format(method))

        fetches = {}
        for fetch_req in fetch_list:
            try:
                if isinstance(fetch_req, tuple):
                    fetches[fetch_req[1]] = getattr(self.video_model, fetch_req[0])[fetch_req[1]]
                else:
                    fetches[fetch_req] = getattr(self.video_model, fetch_req)
            except Exception as err:
                print("%{0}: Failed to retrieve {1} from video_model-attribute.".format(method, fetch_req))
                raise err

        return fetches

    def write_to_summary(self):
        self.summary_writer.add_summary(self.results["summary_op"], self.results["global_step"])
        self.summary_writer.add_summary(self.val_results["summary_op"], self.results["global_step"])
        self.summary_writer.flush()

    def print_results(self, step, results):
        """
        Print the training results /validation results from the training step.
        """
        method = TrainModel.print_results.__name__
        train_epoch = step/self.steps_per_epoch
        print("%{0}: Progress global step {1:d}  epoch {2:.1f}".format(method, step + 1, train_epoch))
        if self.video_model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel":
            print ("Total_loss:{}".format(results["total_loss"]))
        elif self.video_model.__class__.__name__ == "ConvLstmGANVideoPredictionModel":
            print("Total_loss:{}"
                  .format(results["total_loss"]))
        else:
            print("Total_loss:{}"
                  .format(results["total_loss"]))
    @staticmethod
    def plot_train(train_losses, val_losses, loss_name, output_dir):
        """
        Function to plot training losses for train and val datasets against steps
        params:
            train_losses/val_losses       : list, train losses, which length should be equal to the number of training steps
            step                          : int, current training step
            output_dir                    : str,  the path to save the plot
        """ 
   
        iterations = list(range(len(train_losses)))
        if len(train_losses) != len(val_losses): 
            raise ValueError("The length of training losses must be equal to the length of val losses!")  
        plt.plot(iterations, train_losses, 'g', label='Training loss')
        plt.plot(iterations, val_losses, 'b', label='validation loss')
        plt.ylim(10**-5, 10**2)
        plt.yscale("log")
        plt.title('Training and Validation loss')
        plt.xlabel('Iterations')
        plt.ylabel(loss_name)
        plt.legend()
        plt.savefig(os.path.join(output_dir,'plot_train.png'))
        plt.close()

    @staticmethod
    def save_results_to_dict(results_dict,output_dir):
        with open(os.path.join(output_dir,"results.json"),"w") as fp:
            json.dump(results_dict,fp) 

    @staticmethod
    def save_results_to_pkl(train_losses,val_losses, output_dir):
         with open(os.path.join(output_dir,"train_losses.pkl"),"wb") as f:
            pkl.dump(train_losses,f)
         with open(os.path.join(output_dir,"val_losses.pkl"),"wb") as f:
            pkl.dump(val_losses,f) 

    @staticmethod
    def save_timing_to_pkl(total_time,training_time,time_per_iteration, output_dir):
         with open(os.path.join(output_dir,"timing_total_time.pkl"),"wb") as f:
            pkl.dump(total_time,f)
         with open(os.path.join(output_dir,"timing_training_time.pkl"),"wb") as f:
            pkl.dump(training_time,f)
         with open(os.path.join(output_dir,"timing_per_iteration_time.pkl"),"wb") as f:
            pkl.dump(time_per_iteration,f)
    
    @staticmethod        
    def save_loss_per_iteration_to_pkl(loss_per_iteration_train,loss_per_iteration_val, output_dir):
        with open(os.path.join(output_dir,"loss_per_iteration_train.pkl"),"wb") as f:
            pkl.dump(loss_per_iteration_train,f)
        with open(os.path.join(output_dir,"loss_per_iteration_val.pkl"),"wb") as f:
            pkl.dump(loss_per_iteration_val,f)



class BestModelSelector(object):
    """
    Class to select the best performing model from multiple checkpoints created during training
    """
        
    def __init__(self, model_dir: str, eval_metric: str, ltest: bool, criterion: str = "min", channel: int = 0,
                 seed: int = 42):
        """
        Class to retrieve the best model checkpoint. The last one is also retained.
        :param model_dir: path to directory where checkpoints are saved (the trained model output directory)
        :param eval_metric: evaluation metric for model selection (must be implemented in Scores)
        :param criterion: set to 'min' ('max') for negatively (positively) oriented metrics
        :param channel: channel of data used for selection
        :param seed: seed for the Postprocess-instance
        :param ltest: flag to allow bootstrapping in Postprocessing on tiny datasets
        """
        method = self.__class__.__name__
        # sanity check
        if not os.path.isdir(model_dir):
            raise NotADirectoryError("{0}: The passed directory '{1}' does not exist".format(method, model_dir))
        assert criterion in ["min", "max"], "%{0}: criterion must be either 'min' or 'max'.".format(method)
        # set class attributes
        self.seed = seed
        self.channel = channel
        self.metric = eval_metric
        self.checkpoint_base_dir = model_dir
        self.ltest = ltest
        self.checkpoints_all = BestModelSelector.get_checkpoints_dirs(model_dir)
        self.ncheckpoints = len(self.checkpoints_all)
        # evaluate all checkpoints...
        self.checkpoints_eval_all = self.run(self.metric)
        # ... and finalize by choosing the best model and cleaning up
        _ = self.finalize(criterion)

    def run(self, eval_metric):
        """
        Runs eager postprocessing on all checkpoints with evaluation of chosen metric
        :param eval_metric: the target evaluation metric
        :return: Populated self.checkpoints_eval_all where the average of the metric over all forecast hours is listed

        """
        method = BestModelSelector.run.__name__
        from main_visualize_postprocess import Postprocess
        metric_avg_all = []

        for checkpoint in self.checkpoints_all:
            print("Start to evalute checkpoint:", checkpoint)
            results_dir_eager = os.path.join(checkpoint, "results_eager")
            eager_eval = Postprocess(results_dir=results_dir_eager, checkpoint=checkpoint, data_mode="val", batch_size=32,
                                     seed=self.seed, eval_metrics=[eval_metric], channel=self.channel, frac_data=0.33,
                                     lquick=True, ltest=self.ltest)
            eager_eval.run()
            eager_eval.handle_eval_metrics()

            eval_metric_ds = eager_eval.eval_metrics_ds

            metric_avg_all.append(BestModelSelector.get_avg_var(eval_metric_ds, "avg"))
            print("Checkpoint {} is evaluated".format(checkpoint))

        return metric_avg_all

    def finalize(self, criterion):
        """
        Choose the best performing model checkpoint and delete all checkpoints apart from the best and the final ones
        :return: status if everything runs
        """
        method = BestModelSelector.finalize.__name__

        best_ind = self.get_best_checkpoint(criterion)
        if best_ind == self.ncheckpoints -1:
            print("%{0}: Last model checkpoint performs best ({1}: {2:.5f}) and is retained exclusively."
                  .format(method, self.metric, self.checkpoints_eval_all[-1]))
        else:
            print("%{0}: The last ({1}: {2:.5f}) and the best ({1}: {3:.5f}) model checkpoint are retained."
                  .format(method, self.metric, self.checkpoints_eval_all[-1], self.checkpoints_eval_all[best_ind]))

        stat = self.clean_checkpoints(best_ind)
        return stat

    def get_best_checkpoint(self, criterion: str):
        """
        Choose the best performing model checkpoint
        :param criterion: "max" or "min"
        :return: index of best checkpoint in terms of evaluation metric
        """
        method = BestModelSelector.get_best_checkpoint.__name__

        if not self.checkpoints_eval_all:
            raise AttributeError("%{0}: checkpoints_eval_all is still empty. run-method must be executed beforehand."
                                 .format(method))

        if criterion == "min":
            best_index = np.argmin(self.checkpoints_eval_all)
        else:
            best_index = np.argmax(self.checkpoints_eval_all)

        return best_index

    def clean_checkpoints(self, best_ind: int):
        """
        Delete all checkpoints apart from the best and the final ones
        :param best_ind: index of best performing checkpoint
        :return: status
        """
        method = BestModelSelector.clean_checkpoints.__name__

        # list of checkpoints to keep (while ensuring uniqueness!)
        checkpoints_keep = list({self.checkpoints_all[best_ind], self.checkpoints_all[-1]})
        print("%{0}: The following checkpoints are retained: \n * {1}".format(method, "\n* ".join(checkpoints_keep)))
        # drop checkpoints of interest from removal-list
        checkpoints_op = self.checkpoints_all.copy()
        for keep in checkpoints_keep:
            checkpoints_op.remove(keep)

        for dir_path in checkpoints_op:
            shutil.rmtree(dir_path)
            print("%{0}: The checkpoint directory {1} was removed.".format(method, dir_path))

        return True

    @staticmethod
    def get_checkpoints_dirs(model_dir):
        """
        Function to obtain all checkpoint directories in a list.
        :param model_dir: path to directory where checkpoints are saved (the trained model output directory)
        :return: list of all checkpoint directories in model_dir
        """
        method = BestModelSelector.get_checkpoints_dirs.__name__

        checkpoints_all = glob.glob(os.path.join(model_dir, "checkpoint*/"))
        ncheckpoints = len(checkpoints_all)
        if ncheckpoints == 0:
            raise FileExistsError("{0}: No checkpoint folders found under '{1}'".format(method, model_dir))
        else:
            # glob.glob yiels unsorted directories, i.e. do the soring now
            checkpoints_all = sorted(checkpoints_all, key=lambda x: int(x.split("_")[-1].replace("/","")))
            print("%{0}: {1:d} checkpoints directories has been found.".format(method, ncheckpoints))

        return checkpoints_all

    @staticmethod
    def get_avg_var(ds: xr.Dataset, varname_substr: str):
        """
        Retrieves and averages variable from dataset
        :param ds: the dataset
        :param varname_substr: the name of the variable or a substring suifficient to retrieve the variable
        :return: the averaged variable
        """
        varnames = list(ds.variables)
        var_in_file = [s for s in varnames if varname_substr in s]
        try:
            var_mean = ds[var_in_file[0]].mean().values
        except Exception as err:
            raise err

        return var_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory where input data as TFRecord-files are stored.")
    parser.add_argument("--output_dir", help="Output directory where JSON-files, summary, model, plots etc. are saved.")
    parser.add_argument("--datasplit_dict", help="JSON-file that contains the datasplit configuration")
    parser.add_argument("--checkpoint", help="Checkpoint directory or checkpoint name (e.g. <my_dir>/model-200000)")
    parser.add_argument("--dataset", type=str, help="Dataset name") # as in dataset_utils.DATASETS
    parser.add_argument("--model", type=str, help="Model class name")
    parser.add_argument("--model_hparams_dict", type=str, help="JSON-file of model hyperparameters")
    parser.add_argument("--gpu_mem_frac", type=float, default=0.99, help="Fraction of gpu memory to use")
    parser.add_argument("--frac_start_save", type=float, default=1.,
                        help="Fraction of all iteration steps after which checkpointing starts.")
    parser.add_argument("--frac_intv_save", type=float, default=0.01,
                        help="Fraction of all iteration steps to define the saving interval.")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--test_mode", "-test", dest="test_mode", default=False, action="store_true",
                        help="Test mode for postprocessing to allow bootstrapping on small datasets.")

    args = parser.parse_args()
    # start timing for the whole run

    # list pip environment
    import os
    print(os.system("pip3 list"))

    timeit_start = time.time()
    # create a training instance
    train_case = TrainModel(input_dir=args.input_dir,output_dir=args.output_dir,datasplit_dict=args.datasplit_dict,
                 model_hparams_dict=args.model_hparams_dict,model=args.model,checkpoint=args.checkpoint, dataset_name=args.dataset,
                 gpu_mem_frac=args.gpu_mem_frac, seed=args.seed, args=args, frac_start_save=args.frac_start_save,
                 frac_intv_save=args.frac_intv_save)
    
    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')
    
    # setup
    train_case.setup() 
 
    # train model
    train_time, time_per_iteration = train_case.train_model()
    timeit_after_train = time.time()
    train_case.save_timing_to_pkl(timeit_after_train - timeit_start, train_time, time_per_iteration, args.output_dir)

    # select best model
    if args.dataset == "era5" and args.frac_start_save < 1.:
        _ = BestModelSelector(args.output_dir, "mse", args.test_mode)
        timeit_finish = time.time()
        print("Selecting the best model checkpoint took {0:.2f} minutes.".format((timeit_finish - timeit_after_train)/60.))
    else:
        timeit_finish = time.time()
    print("Total time elapsed {0} minutes.".format((timeit_finish - timeit_start)/60.))


if __name__ == '__main__':
    main()
