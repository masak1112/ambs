from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
We took the code implementation from https://github.com/alexlee-gk/video_prediction, SAVP model  as reference, and adjust the code based on our project needs
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-10-22"

import argparse
import errno
import json
import os
import random
import time
import numpy as np
import tensorflow as tf
from model_modules.video_prediction import datasets, models
import matplotlib.pyplot as plt
import pickle as pkl
from model_modules.video_prediction.utils import tf_utils


class TrainModel(object):
    def __init__(self, input_dir: str = None, output_dir: str = None, datasplit_dict: str = None,
                 model_hparams_dict: str = None, model: str = None, checkpoint: str = None, dataset: str = None,
                 gpu_mem_frac: float = None, seed: int = None, args=None, save_diag_intv: int = 100,
                 niter_loss_avg: int = 10000):
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
        :param save_diag_intv: interval of iteration steps for which diagnostic plot and optional model saving is done
        """
        self.input_dir = os.path.normpath(input_dir)
        self.output_dir = os.path.normpath(output_dir)
        self.datasplit_dict = datasplit_dict
        self.model_hparams_dict = model_hparams_dict
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.model = model
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.args = args
        # for diagnozing and saving the model during training
        self.save_diag_intv = save_diag_intv
        self.niter_loss_avg = niter_loss_avg

    def setup(self):
        self.set_seed()
        self.get_model_hparams_dict()
        self.load_params_from_checkpoints_dir()
        self.setup_dataset()
        self.setup_model()
        self.make_dataset_iterator()
        self.setup_graph()
        self.save_dataset_model_params_to_checkpoint_dir(dataset=self.train_dataset,video_model=self.video_model)
        self.count_parameters()
        self.create_saver_and_writer()
        self.setup_gpu_config()
        self.calculate_samples_and_epochs()

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
        if self.output_dir is None:
            raise ValueError("Output_dir-argument is empty. Please define a proper output_dir")
        elif not os.path.isdir(self.output_dir):
            raise NotADirectoryError("Base output_dir {0} does not exist. Please pass a proper output_dir and "+\
                                     "make use of config_train.py.")

            
    def get_model_hparams_dict(self):
        """
        Get and read model_hparams_dict from json file to dictionary 
        """
        self.model_hparams_dict_load = {}
        if self.model_hparams_dict:
            with open(self.model_hparams_dict) as f:
                self.model_hparams_dict_load.update(json.loads(f.read()))
        return self.model_hparams_dict_load


    def load_params_from_checkpoints_dir(self):
        """
        If checkpoint is none, load and read the json files of datasplit_config, and hparam_config,
        and use the corresponding parameters.
        If the checkpoint is given, the configuration of dataset, model and options in the checkpoint dir will be
        restored and used for continue training.
        """
        if self.checkpoint:
            self.checkpoint_dir = os.path.normpath(self.checkpoint)
            if not os.path.isdir(self.checkpoint):
                self.checkpoint_dir, _ = os.path.split(self.checkpoint_dir)
            if not os.path.exists(self.checkpoint_dir):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.checkpoint_dir)
            # read and overwrite dataset and model from checkpoint
            try:
                with open(os.path.join(self.checkpoint_dir, "options.json")) as f:
                    print("loading options from checkpoint %s" % self.checkpoint)
                    self.options = json.loads(f.read())
                    self.dataset = self.dataset or self.options['dataset']
                    self.model = self.model or self.options['model']
            except FileNotFoundError:
                print("options.json was not loaded because it does not exist in {0}".format(self.checkpoint_dir))
            # loading hyperparameters from checkpoint
            try:
                with open(os.path.join(self.checkpoint_dir, "model_hparams.json")) as f:
                    self.model_hparams_dict_load.update(json.loads(f.read()))
            except FileNotFoundError:
                print("model_hparams.json was not loaded because it does not exist in {0}".format(self.checkpoint_dir))
                
    def setup_dataset(self):
        """
        Setup train and val dataset instance with the corresponding data split configuration.
        Simultaneously, sequence_length is attached to the hyperparameter dictionary.
        """
        self.train_dataset = VideoDataset(input_dir=self.input_dir, mode='train', datasplit_config=self.datasplit_dict,
                                          hparams_dict_config=self.model_hparams_dict)
        self.val_dataset = VideoDataset(input_dir=self.input_dir, mode='val', datasplit_config=self.datasplit_dict,
                                        hparams_dict_config=self.model_hparams_dict)
        # ML/BG 2021-06-15: Is the following needed?
        # self.model_hparams_dict_load.update({"sequence_length": self.train_dataset.sequence_length})

    def setup_model(self):
        """
        Set up model instance for the given model names
        """
        VideoPredictionModel = models.get_model_class(self.model)
        self.video_model = VideoPredictionModel(hparams_dict=self.model_hparams_dict_load)

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
        self.train_tf_dataset = self.train_dataset.make_dataset(self.batch_size)
        self.train_iterator = self.train_tf_dataset.make_one_shot_iterator()
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.train_handle = self.train_iterator.string_handle()
        self.val_tf_dataset = self.val_dataset.make_dataset(self.batch_size)
        self.val_iterator = self.val_tf_dataset.make_one_shot_iterator()
        self.val_handle = self.val_iterator.string_handle()
        self.iterator = tf.data.Iterator.from_string_handle(
            self.train_handle, self.train_tf_dataset.output_types, self.train_tf_dataset.output_shapes)
        self.inputs = self.iterator.get_next()
        #since era5 tfrecords include T_start, we need to remove it from the tfrecord when we train the model,
        # otherwise the model will raise error
        
        if self.dataset == "era5" and self.model == "savp":
           del self.inputs["T_start"]


    def save_dataset_model_params_to_checkpoint_dir(self, dataset, video_model):
        """
        Save all setup configurations such as args, data_hparams, and model_hparams into output directory
        """
        with open(os.path.join(self.output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(self.args), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "dataset_hparams.json"), "w") as f:
            f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "model_hparams.json"), "w") as f:
            f.write(json.dumps(video_model.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "data_dict.json"), "w") as f:
            f.write(json.dumps(dataset.data_dict, sort_keys=True, indent=4))


    def count_parameters(self):
        """
        Count the paramteres of the model
        """ 
        with tf.name_scope("parameter_count"):
            # exclude trainable variables that are replicas (used in multi-gpu setting)
            self.trainable_variables = set(tf.trainable_variables()) & set(self.video_model.saveable_variables)
            self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in self.trainable_variables])


    def create_saver_and_writer(self):
        """
        Create saver to save the models latest checkpoints, and a summery writer to store the train/val metrics  
        """
        self.saver = tf.train.Saver(var_list=self.video_model.saveable_variables, max_to_keep=2)
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

        batch_size = self.video_model.hparams.batch_size
        max_epochs = self.video_model.hparams.max_epochs   #the number of epochs
        self.num_examples = self.train_dataset.num_examples_per_epoch()
        self.steps_per_epoch = int(self.num_examples/batch_size)
        self.total_steps = self.steps_per_epoch * max_epochs
        print("%{}: Batch size: {}; max_epochs: {}; num_samples per epoch: {}; steps_per_epoch: {}, total steps: {}"
              .format(method, batch_size, max_epochs, self.num_examples, self.steps_per_epoch, self.total_steps))

    def restore(self, sess, checkpoints, restore_to_checkpoint_mapping=None):
        """
        Restore the models checkpoints if the checkpoints is given
        """
        method = TrainModel.restore.__name__

        if checkpoints is None:
            print("%{0}: Checkpoint-variable is not set!".format(method))
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
                print("%{0}: Creating restore saver from checkpoint {1}".format(method, checkpoint))
                saver, _ = tf_utils.get_checkpoint_restore_saver(
                   checkpoint, var_list, skip_global_step=skip_global_step,
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
        elif os.path.isdir(self.checkpoint) and (not os.path.exists(os.path.join(self.output_dir,"checkpoint"))):
            train_losses,val_losses = [], []
        else:
            with open(os.path.join(self.output_dir,"train_losses.pkl"),"rb") as f:
                train_losses = pkl.load(f)
            with open(os.path.join(self.output_dir,"val_losses.pkl"),"rb") as f:
                val_losses = pkl.load(f)
        return train_losses,val_losses

    def train_model(self):
        """
        Start session and train the model by looping over all iteration steps
        """
        method = TrainModel.train_model.__name__

        self.global_step = tf.train.get_or_create_global_step()

        with tf.Session(config=self.config) as sess:
            print("parameter_count =", sess.run(self.parameter_count))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.restore(sess, self.checkpoint)
            start_step = sess.run(self.global_step)
            print("Iteration starts at step {0}".format(start_step))
            # start at one step earlier to log everything without doing any training
            # step is relative to the start_step
            train_losses, val_losses = self.restore_train_val_losses()
            # initialize auxiliary variables
            time_per_iteration = []
            run_start_time = time.time()
            val_loss_min = 999.
            # perform iteration
            for step in range(start_step, self.total_steps):
                timeit_start = time.time()
                # run for training dataset
                self.create_fetches_for_train()             # In addition to the loss, we fetch the optimizer
                self.results = sess.run(self.fetches)       # ...and run it here!
                train_losses.append(self.results["total_loss"])
                # run and fetch losses for validation data
                val_handle_eval = sess.run(self.val_handle)
                self.create_fetches_for_val()
                self.val_results = sess.run(self.val_fetches,feed_dict={self.train_handle: val_handle_eval})
                val_losses.append(self.val_results["total_loss"])
                self.write_to_summary()
                self.print_results(step,self.results)
                # track iteration time
                time_iter = time.time() - timeit_start
                time_per_iteration.append(time_iter)
                print("time needed for this step {0:.3f}s".format(time_iter))
                if step % self.save_diag_intv == 0:
                    lsave, val_loss_min = TrainModel.set_model_saver_flag(val_losses, val_loss_min, self.niter_loss_avg)
                    if lsave:
                        self.saver.save(sess, os.path.join(self.output_dir, "model"), global_step=step)
                    # pickle file and plots are always created
                    TrainModel.save_results_to_pkl(train_losses,val_losses,self.output_dir)
                    TrainModel.plot_train(train_losses,val_losses,step,self.output_dir)

            # Final diagnostics
            # track time (save to pickle-files)
            train_time = time.time() - run_start_time
            results_dict = {"train_time": train_time,
                            "total_steps": self.total_steps}
            TrainModel.save_results_to_dict(results_dict,self.output_dir)

            avg_samples = int(2000)
            print("%{0}: Training loss decreased from {1:.6f} to {2:.6f}:"
                  .format(method, np.mean(train_losses[0:10]), np.mean(train_losses[-avg_samples:])))
            print("%{0}: Validation loss decreased from {1:.6f} to {2:.6f}:"
                  .format(method, np.mean(val_losses[0:10]), np.mean(val_losses[-avg_samples:])))
            print("%{0}: Training finsished".format(method))
            print("%{0}: Total training time: {1:.2f} min".format(method, train_time/60.))

            return train_time, time_per_iteration
            
 
    def create_fetches_for_train(self):
       """
       Fetch variables in the graph, this can be custermized based on models and also the needs of users
       """
       #This is the base fetch that for all the  models
       self.fetches = {"train_op": self.video_model.train_op}         # fetching the optimizer!
       self.fetches["summary"] = self.video_model.summary_op
       self.fetches["global_step"] = self.global_step
       if self.video_model.__class__.__name__ == "McNetVideoPredictionModel": self.fetches_for_train_mcnet()
       if self.video_model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel": self.fetches_for_train_convLSTM()
       if self.video_model.__class__.__name__ == "SAVPVideoPredictionModel": self.fetches_for_train_savp()
       if self.video_model.__class__.__name__ == "VanillaVAEVideoPredictionModel": self.fetches_for_train_vae()
       if self.video_model.__class__.__name__ == "VanillaGANVideoPredictionModel":self.fetches_for_train_gan()
       if self.video_model.__class__.__name__ == "ConvLstmGANVideoPredictionModel":self.fetches_for_train_convLSTM()
       return self.fetches     
    
    def fetches_for_train_convLSTM(self):
        """
        Fetch variables in the graph for convLSTM model, this can be custermized based on models and the needs of users
        """
        self.fetches["total_loss"] = self.video_model.total_loss
        self.fetches["inputs"] = self.video_model.inputs

 
    def fetches_for_train_savp(self):
        """
        Fetch variables in the graph for savp model, this can be custermized based on models and the needs of users
        """
        self.fetches["g_losses"] = self.video_model.g_losses
        self.fetches["d_losses"] = self.video_model.d_losses
        self.fetches["d_loss"] = self.video_model.d_loss
        self.fetches["g_loss"] = self.video_model.g_loss
        self.fetches["total_loss"] = self.video_model.g_loss
        self.fetches["inputs"] = self.video_model.inputs


    def fetches_for_train_mcnet(self):
        """
        Fetch variables in the graph for mcnet model, this can be custermized based on models and  the needs of users
        """
        self.fetches["L_p"] = self.video_model.L_p
        self.fetches["L_gdl"] = self.video_model.L_gdl
        self.fetches["L_GAN"]  = self.video_model.L_GAN        

    def fetches_for_train_vae(self):
        """
        Fetch variables in the graph for savp model, this can be custermized based on models and based on the needs of users
        """
        self.fetches["latent_loss"] = self.video_model.latent_loss
        self.fetches["recon_loss"] = self.video_model.recon_loss
        self.fetches["total_loss"] = self.video_model.total_loss

    def fetches_for_train_gan(self):
        self.fetches["total_loss"] = self.video_model.total_loss

    def create_fetches_for_val(self):
        """
        Fetch variables in the graph for validation dataset, this can be custermized based on models and the needs of users
        """
        if self.video_model.__class__.__name__ == "SAVPVideoPredictionModel":
            self.val_fetches = {"total_loss": self.video_model.g_loss}
            self.val_fetches["inputs"] = self.video_model.inputs
        else:
            self.val_fetches = {"total_loss": self.video_model.total_loss}
            self.val_fetches["inputs"] = self.video_model.inputs
        self.val_fetches["summary"] = self.video_model.summary_op

    def write_to_summary(self):
        self.summary_writer.add_summary(self.results["summary"],self.results["global_step"])
        self.summary_writer.add_summary(self.val_results["summary"],self.results["global_step"])
        self.summary_writer.flush()

    def print_results(self, step, results):
        """
        Print the training results /validation results from the training step.
        """
        method = TrainModel.print_results.__name__

        train_epoch = step/self.steps_per_epoch
        print("progress  global step %d  epoch %0.1f" % (step + 1, train_epoch))
        if self.video_model.__class__.__name__ == "McNetVideoPredictionModel":
            print("Total_loss:{}; L_p_loss:{}; L_gdl:{}; L_GAN: {}".format(results["total_loss"],results["L_p"],
                                                                           results["L_gdl"],results["L_GAN"]))
        elif self.video_model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel":
            print ("Total_loss:{}".format(results["total_loss"]))
        elif self.video_model.__class__.__name__ == "SAVPVideoPredictionModel":
            print("Total_loss/g_losses:{}; d_losses:{}; g_loss:{}; d_loss: {}"
                  .format(results["g_losses"], results["d_losses"], results["g_loss"], results["d_loss"]))
        elif self.video_model.__class__.__name__ == "VanillaVAEVideoPredictionModel":
            print("Total_loss:{}; latent_losses:{}; reconst_loss:{}".format(results["total_loss"],
                                                                            results["latent_loss"],
                                                                            results["recon_loss"]))
        else:
            print("%{0}: WARNING: The model name does not exist".format(method))
    

    @staticmethod
    def set_model_saver_flag(losses: list, old_min_loss: float, niter_steps: int = 100):
        """
        Sets flag to save the model given that a new minimum in the loss is readched
        :param losses: list of losses over iteration steps
        :param old_min_loss: previous loss
        :param niter_steps: number of iteration steps over which the loss is averaged
        :return flag: True if model should be saved
        :return loss_avg: updated minimum loss
        """

        save_flag = False
        if len(losses) <= niter_steps*2:
            loss_avg = old_min_loss
            return save_flag, loss_avg

        loss_avg = np.mean(losses[-niter_steps:])
        if loss_avg < old_min_loss:
            save_flag = True
        else:
            loss_avg = old_min_loss

        return save_flag, loss_avg


    @staticmethod
    def plot_train(train_losses, val_losses, step, output_dir):
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
        plt.ylabel('Loss')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams")
    parser.add_argument("--datasplit_dict", help="json file that contains the datasplit configuration")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")
    parser.add_argument("--gpu_mem_frac", type=float, default=0.99, help="fraction of gpu memory to use")
    parser.add_argument("--seed",default=1234, type=int)
    args = parser.parse_args()
    # start timing for the whole run
    timeit_start_total_time = time.time()  
    #create a training instance
    train_case = TrainModel(input_dir=args.input_dir,output_dir=args.output_dir,datasplit_dict=args.datasplit_dict,
                 model_hparams_dict=args.model_hparams_dict,model=args.model,checkpoint=args.checkpoint,dataset=args.dataset,
                 gpu_mem_frac=args.gpu_mem_frac,seed=args.seed,args=args)  
    
    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')
    
    # setup
    train_case.setup() 
 
    # train model
    train_time, time_per_iteration = train_case.train_model()
       
    total_run_time = time.time() - timeit_start_total_time
    train_case.save_timing_to_pkl(total_run_time, train_time, time_per_iteration, args.output_dir)
    
if __name__ == '__main__':
    main()
