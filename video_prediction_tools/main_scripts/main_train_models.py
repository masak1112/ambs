from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
We took the code implementation from https://github.com/alexlee-gk/video_prediction, SAVP model  as reference, and adjust the code based on our project needs
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler, Michael Langguth"
__date__ = "2020-10-22"

import argparse
import errno
import json
import os
import random
import time
import numpy as np
import tensorflow as tf
from video_prediction import datasets, models
import matplotlib.pyplot as plt
import pickle as pkl
from video_prediction.utils import tf_utils


class TrainModel(object):
    def __init__(self, input_dir=None, output_dir=None, datasplit_dict=None,
                       model_hparams_dict=None, model=None,
                       checkpoint=None, dataset=None,
                       gpu_mem_frac=None, seed=None, args=None, save_interval=20):
        
        """
        This class aims to train the models
        args:
            input_dir            : str, the path to the PreprocessData directory which is parent directory of "Pickle" and "tfrecords" files directiory. 
            output_dir           : str, directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams
            datasplit_dict       : str, the path pointing to the datasplit_config json file
            model_hparams_dict   : str, a json file of model hyperparameters
            checkpoint           : str, directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)
            dataset              : str, dataset class name
            model                : str, model class name
            gpu_mem_frac         : float, fraction of gpu memory to use
            save_interval        :int, how many steps for saving the train/val loss be saved
        """ 
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.datasplit_dict = datasplit_dict
        self.model_hparams_dict = model_hparams_dict
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.model = model
        self.gpu_mem_frac = gpu_mem_frac
        self.seed = seed
        self.args = args
        self.save_interval = save_interval
        self.generate_output_dir()

    def setup(self):
        self.set_seed()
        self.load_params_from_checkpoints_dir()
        self.setup_dataset()
        self.setup_model()
        self.make_dataset_iterator()
        self.setup_graph()
        self.save_dataset_model_params_to_checkpoint_dir()
        self.count_parameters()
        self.create_saver_and_writer()
        self.setup_gpu_config()
        self.calculate_samples_and_epochs()

    def set_seed(self):
        if self.seed is not None:
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    def generate_output_dir(self):
        if self.output_dir is None:
            raise ("Output_dir is None, Please define the proper output_dir")

            
    def get_model_hparams_dict(self):
        """
        Get model_hparams_dict from json file
        """
        self.model_hparams_dict_load = {}
        if self.model_hparams_dict:
            with open(self.model_hparams_dict) as f:
                self.model_hparams_dict_load.update(json.loads(f.read()))
        return self.model_hparams_dict_load


    def load_params_from_checkpoints_dir(self):
        """
        load the json files related datasets , model configure metadata (This information was stored in the checkpoint dir when last time training model)
        """
        self.get_model_hparams_dict()
        if self.checkpoint:
            self.checkpoint_dir = os.path.normpath(self.checkpoint)
            if not os.path.isdir(self.checkpoint):
                self.checkpoint_dir, _ = os.path.split(self.checkpoint_dir)
            if not os.path.exists(self.checkpoint_dir):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.checkpoint_dir)
            with open(os.path.join(self.checkpoint_dir, "options.json")) as f:
                print("loading options from checkpoint %s" % self.checkpoint)
                self.options = json.loads(f.read())
                self.dataset = self.dataset or options['dataset']
                self.model = self.model or options['model']
            try:
                with open(os.path.join(self.checkpoint_dir, "model_hparams.json")) as f:
                    self.model_hparams_dict_load.update(json.loads(f.read()))
            except FileNotFoundError:
                print("model_hparams.json was not loaded because it does not exist")
                
    def setup_dataset(self):
        """
        Setup train and val dataset instance with the corresponding data split configuration
        """
        VideoDataset = datasets.get_dataset_class(self.dataset)
        self.train_dataset = VideoDataset(input_dir=self.input_dir,mode='train',datasplit_config=self.datasplit_dict)
        self.val_dataset = VideoDataset(input_dir=self.input_dir, mode='val',datasplit_config=self.datasplit_dict)
        #self.variable_scope = tf.get_variable_scope()
        #self.variable_scope.set_use_resource(True)
      

    def setup_model(self):
        """
        Set up model instance
        """
        VideoPredictionModel = models.get_model_class(self.model)
        self.video_model = VideoPredictionModel(
                                    hparams_dict=self.model_hparams_dict_load,
                                       )
    def setup_graph(self):
        """
            build model graph
            since era5 tfrecords include T_start, we need to remove it from the tfrecord when we train the model, otherwise the model will raise error 
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
        if self.dataset == "era5" and self.model == "savp":
           del  self.inputs["T_start"]        


    def save_dataset_model_params_to_checkpoint_dir(self):
        """
        Save all setup configurations such as args, data_hparams, and model_hparams into output directory
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(os.path.join(self.output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(self.args), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "dataset_hparams.json"), "w") as f:
            f.write(json.dumps(self.train_dataset.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "model_hparams.json"), "w") as f:
            f.write(json.dumps(self.video_model.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "data_dict.json"), "w") as f:
            f.write(json.dumps(self.train_dataset.data_dict, sort_keys=True, indent=4))  



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
        self.saver = tf.train.Saver(var_list=self.video_model.saveable_variables, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.output_dir)

    def setup_gpu_config(self):
        """
        Setup GPU options 
        """
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_mem_frac, allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=self.gpu_options, allow_soft_placement=True)


    def calculate_samples_and_epochs(self):
        """
        Clculate the number of samples for train/val/testing dataset. The samples are used for training model for each epoch. 
        Clculate the iterations (samples multiple by max_epochs) for training.
        """
        batch_size = self.video_model.hparams.batch_size
        max_epochs = self.video_model.hparams.max_epochs #the number of epochs
        self.num_examples = self.train_dataset.num_examples_per_epoch()
        self.steps_per_epoch = int(self.num_examples/batch_size)
        self.total_steps = self.steps_per_epoch * max_epochs

    def restore(self,sess, checkpoints, restore_to_checkpoint_mapping=None):
        if checkpoints:
           var_list = self.video_model.saveable_variables
           # possibly restore from multiple checkpoints. useful if subset of weights
           # (e.g. generator or discriminator) are on different checkpoints.
           if not isinstance(checkpoints, (list, tuple)):
               checkpoints = [checkpoints]
           # automatically skip global_step if more than one checkpoint is provided
           skip_global_step = len(checkpoints) > 1
           savers = []
           for checkpoint in checkpoints:
               print("creating restore saver from checkpoint %s" % checkpoint)
               saver, _ = tf_utils.get_checkpoint_restore_saver(
                   checkpoint, var_list, skip_global_step=skip_global_step,
                   restore_to_checkpoint_mapping=restore_to_checkpoint_mapping)
               savers.append(saver)
           restore_op = [saver.saver_def.restore_op_name for saver in savers]
           sess.run(restore_op)
    
    def restore_train_val_losses(self):
        """
        Restore the train and validation losses in the pickle file
        """
        if self.start_step == 0:
            train_losses = []
            val_losses = []
        else:
            with open(os.path.join(self.checkpoint,"train_losses.pkl"),"rb") as f:
                train_losses = pkl.load(f)
            with open(os.path.join(self.checkpoint,"val_losses.pkl"),"rb") as f:
                val_losses = pkl.load(f)
        return train_losses,val_losses

    def train_model(self):
        """
        Start session and train the model
        """
        global_step = tf.train.get_or_create_global_step()
        with tf.Session(config=self.config) as sess:
            print("parameter_count =", sess.run(self.parameter_count))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.restore(sess, self.checkpoint)
            #sess.graph.finalize()
            self.start_step = sess.run(global_step)
            print("start_step", self.start_step)
            # start at one step earlier to log everything without doing any training
            # step is relative to the start_step
            train_losses, val_losses = self.restore_train_val_losses()
            run_start_time = time.time()
            for step in range(self.start_step,self.total_steps):
                timeit_start = time.time()
                #run for training dataset
                self.create_fetches_for_train()
                self.results = sess.run(self.fetches)
                train_losses.append(self.results["total_loss"])
                #Run and fetch losses for validation data
                val_handle_eval = sess.run(self.val_handle)
                self.create_fetches_for_val()
                self.val_results = sess.run(self.val_fetches,feed_dict={self.train_handle: val_handle_eval})
                val_losses.append(self.val_results["total_loss"])
                self.write_to_summary()
                self.print_results(step,self.results)
                self.saver.save(sess, os.path.join(self.output_dir, "model"), global_step=step)
                timeit_end = time.time()
                print("time needed for this step", timeit_end - timeit_start, ' s')
                if step % self.save_interval == 0:
                    # I save the pickle file and plot here inside the loop in case the training process cannot finished after job is done.
                    TrainModel.save_results_to_pkl(train_losses,val_losses,self.output_dir)
                    TrainModel.plot_train(train_losses,val_losses,step,self.output_dir)

            #Totally train time over all the iterations
            train_time = time.time() - run_start_time
            results_dict = {"train_time":train_time,
                            "total_steps":self.total_steps}
            TrainModel.save_results_to_dict(results_dict,self.output_dir)
            print("train_losses:",train_losses)
            print("val_losses:",val_losses) 
            print("Done")
            print("Total training time:", train_time/60., "min")
 
    def create_fetches_for_train(self):
       """
       Fetch variables in the graph, this can be custermized based on models and based on the needs of users
       """
       #This is the base fetch that for all the  models
       self.fetches = {"train_op": self.video_model.train_op}
       self.fetches["summary"] = self.video_model.summary_op
       self.fetches["global_step"] = self.video_model.global_step
       self.fetches["total_loss"] = self.video_model.total_loss
       if self.video_model.__class__.__name__ == "McNetVideoPredictionModel": self.fetches_for_train_mcnet()
       if self.video_model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel": self.fetches_for_train_convLSTM()
       if self.video_model.__class__.__name__ == "SAVPVideoPredictionModel": self.fetches_for_train_savp()
       if self.video_model.__class__.__name__ == "VanillaVAEVideoPredictionModel": self.fetches_for_train_vae()
       return self.fetches     
    
    def fetches_for_train_convLSTM(self):
        """
        Fetch variables in the graph for convLSTM model, this can be custermized based on models and based on the needs of users
        """
        pass

 
    def fetches_for_train_savp(self):
        """
        Fetch variables in the graph for savp model, this can be custermized based on models and based on the needs of users
        """
        self.fetches["g_losses"] = self.video_model.g_losses
        self.fetches["d_losses"] = self.video_model.d_losses
        self.fetches["d_loss"] = self.video_model.d_loss
        self.fetches["g_loss"] = self.video_model.g_loss

    def fetches_for_train_mcnet(self):
        """
        Fetch variables in the graph for mcnet model, this can be custermized based on models and based on the needs of users
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


    def create_fetches_for_val(self):
        """
        Fetch variables in the graph for validation dataset, this can be custermized based on models and based on the needs of users
        """
        self.val_fetches = {"total_loss": self.video_model.total_loss}
        self.val_fetches["summary"] = self.video_model.summary_op

    def write_to_summary(self):
        self.summary_writer.add_summary(self.results["summary"],self.results["global_step"])
        self.summary_writer.add_summary(self.val_results["summary"],self.results["global_step"])
        self.summary_writer.flush()


    def print_results(self,step,results):
        """
        Print the training results /validation results from the training step.
        """
        train_epoch = step/self.steps_per_epoch
        print("progress  global step %d  epoch %0.1f" % (step + 1, train_epoch))
        if self.video_model.__class__.__name__ == "McNetVideoPredictionModel":
            print("Total_loss:{}; L_p_loss:{}; L_gdl:{}; L_GAN: {}".format(results["total_loss"],results["L_p"],results["L_gdl"],results["L_GAN"]))
        elif self.video_model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel":
            print ("Total_loss:{}".format(results["total_loss"]))
        elif self.video_model.__class__.__name__ == "SAVPVideoPredictionModel":
            print("Total_loss/g_losses:{}; d_losses:{}; g_loss:{}; d_loss: {}".format(results["g_losses"],results["d_losses"],results["g_loss"],results["d_loss"]))
        elif self.video_model.__class__.__name__ == "VanillaVAEVideoPredictionModel":
            print("Total_loss:{}; latent_losses:{}; reconst_loss:{}".format(results["total_loss"],results["latent_loss"],results["recon_loss"]))
        else:
            print ("The model name does not exist")


    @staticmethod
    def plot_train(train_losses,val_losses,step,output_dir):
        """
        Function to plot training losses for train and val datasets against steps
        params:
            train_losses/val_losses       :list, train losses, which length should be equal to the number of training steps
            step                          : int, current training step
            output_dir                    : str,  the path to save the plot
        """ 
   
        iterations = list(range(len(train_losses)))
        if len(train_losses) != len(val_losses) or len(train_losses) != step +1 : 
            raise ValueError("The length of training losses must be equal to the length of val losses and  step +1 !")  
        plt.plot(iterations, train_losses, 'g', label='Training loss')
        plt.plot(iterations, val_losses, 'b', label='validation loss')
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
    train_case.train_model()
       
if __name__ == '__main__':
    main()
