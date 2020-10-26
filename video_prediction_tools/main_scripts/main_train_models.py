from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from json import JSONEncoder
import pickle as pkl

class TrainModel(object):
    def __init__(self,input_dir=None,output_dir=None,datasplit_dict=None,
                 model_hparams_dict=None,model=None,
                 checkpoint=None,dataset=None,
                 gpu_mem_frac=None,seed=None):
        
        """
        This class aims to train the models
        args:
            input_dir            : str, the path to the PreprocessData directory which is parent directory of "Pickle" and "tfrecords" files directiory. 
            output_dir           : str, directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams
            datasplit_dict       : str, the path pointing to the datasplit_config json file
            hparams_dict_dict    : str, the path to the dict that contains hparameters,
            dataset              :str, dataset class name
            model                :str, model class name
            model_hparams_dict   :str, a json file of model hyperparameters
            gpu_mem_frac         :float, fraction of gpu memory to use"
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
        self.generate_output_dir()
        self.hparams_dict = self.get_model_hparams_dict()


    def __call__(self):
        self.setup()
        self.train()
        self.plot()
    
    def setup(self):
        self.set_seed()
        self.resume_checkpoint()
        self.load_params_from_checkpoints_dir()
        self.setup_dataset()
        self.setup_model()
        self.make_dataset_iterator()

    def set_seed(self):
        if self.seed is not None:
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)


    def generate_output_dir(self):
        if self.output_dir is None:
            raise ("Output_dir is None, Please define the proper output_dir")
        else:
           self.ouput_dir = os.path.join(self.output_dir,self.model)


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
         load the existing checkpoint related datasets, model configure (This information was stored in the checkpoint dir when last time training model)
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
            self.dataset = dataset or options['dataset']
            self.model = model or options['model']
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
        self.model = VideoPredictionModel(
                                    hparams_dict=self.hparams_dict,
                                       )
    def setup_graph(self):
        self.model.build_graph(self.inputs)                                    

    def make_dataset_iterator(self):
        """
        Prepare the dataset interator for training and validation
        """
        self.batch_size = self.hparams_dict["batch_size"]
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

    def save_dataset_model_params_to_checkpoint_dir(self,args):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(os.path.join(self.output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(args), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "dataset_hparams.json"), "w") as f:
            f.write(json.dumps(train_dataset.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(self.output_dir, "model_hparams.json"), "w") as f:
            f.write(json.dumps(self.model.hparams.values(), sort_keys=True, indent=4))
   
    @staticmethod
    def plot_train(train_losses,val_losses,step,output_dir):
        """
        Function to plot training losses for train and val datasets against steps
        params:
            train_losses/val_losses (list): train losses, which length should be equal to the number of training steps
            step (int): current training step
            output_dir (str): the path to save the plot
    
        """ 
   
        iterations = list(range(len(train_losses)))
        if len(train_losses) != len(val_losses) or len(train_losses) != step +1 : raise ValueError("The length of training losses must be equal to the length of val losses and  step +1 !")  
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
                 gpu_mem_frac=args.gpu_mem_frac,seed=args.seed)  
    
    # setup
    train_case.setup() 
 
     
    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')
    

    batch_size = model.hparams.batch_size
    #Create input and val iterator
    inputs, train_handle, val_handle = make_dataset_iterator(train_dataset, val_dataset, batch_size)
    
    #build model graph
    #since era5 tfrecords include T_start, we need to remove it from the tfrecord when we train the model, otherwise the model will raise error 
    if args.dataset == "era5":
       del inputs["T_start"]
    model.build_graph(inputs)
    
    #save all the model, data params to output dirctory
    save_dataset_model_params_to_checkpoint_dir(args,args.output_dir,train_dataset,model)
    
    with tf.name_scope("parameter_count"):
        # exclude trainable variables that are replicas (used in multi-gpu setting)
        trainable_variables = set(tf.trainable_variables()) & set(model.saveable_variables)
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in trainable_variables])

    saver = tf.train.Saver(var_list=model.saveable_variables, max_to_keep=2)

    # None has the special meaning of evaluating at the end, so explicitly check for non-equality to zero
    summary_writer = tf.summary.FileWriter(args.output_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
 
    max_epochs = model.hparams.max_epochs #the number of epochs
    num_examples_per_epoch = train_dataset.num_examples_per_epoch()
    print ("number of exmaples per epoch:",num_examples_per_epoch)
    steps_per_epoch = int(num_examples_per_epoch/batch_size)
    #number of steps totally equal to the number of steps per each echo multiple by number of epochs
    total_steps = steps_per_epoch * max_epochs
    global_step = tf.train.get_or_create_global_step()
    #mock total_steps only for fast debugging
    #total_steps = 10
    print ("Total steps for training:",total_steps)
    results_dict = {}
    with tf.Session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.restore(sess, args.checkpoint)
        sess.graph.finalize()
        #start_step = sess.run(model.global_step)
        start_step = sess.run(global_step)
        print("start_step", start_step)
        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        train_losses=[]
        val_losses=[]
        run_start_time = time.time()        
        for step in range(start_step,total_steps):
            #global_step = sess.run(global_step)
            # +++ Scarlet 20200813
            timeit_start = time.time()  
            # --- Scarlet 20200813
            print ("step:", step)
            val_handle_eval = sess.run(val_handle)
            #Fetch variables in the graph
            fetches = {"train_op": model.train_op}
            #fetches["latent_loss"] = model.latent_loss
            fetches["summary"] = model.summary_op 
            fetches["global_step"] = model.global_step

            if model.__class__.__name__ == "McNetVideoPredictionModel" or model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel" or model.__class__.__name__ == "VanillaVAEVideoPredictionModel":
                fetches["global_step"] = model.global_step
                fetches["total_loss"] = model.total_loss
                #fetch the specific loss function only for mcnet
                if model.__class__.__name__ == "McNetVideoPredictionModel":
                    fetches["L_p"] = model.L_p
                    fetches["L_gdl"] = model.L_gdl
                    fetches["L_GAN"]  =model.L_GAN
                if model.__class__.__name__ == "VanillaVAEVideoPredictionModel":
                    fetches["latent_loss"] = model.latent_loss
                    fetches["recon_loss"] = model.recon_loss
                results = sess.run(fetches)
                train_losses.append(results["total_loss"])
                #Fetch losses for validation data
                val_fetches = {}
                #val_fetches["latent_loss"] = model.latent_loss
            summary_writer.add_summary(results["summary"],results["global_step"])
            summary_writer.add_summary(val_results["summary"],results["global_step"])
            summary_writer.flush()

            # global_step will have the correct step count if we resume from a checkpoint
            # global step is read before it's incemented
            train_epoch = step/steps_per_epoch
            print("progress  global step %d  epoch %0.1f" % (step + 1, train_epoch))
            if model.__class__.__name__ == "McNetVideoPredictionModel":
                print("Total_loss:{}; L_p_loss:{}; L_gdl:{}; L_GAN: {}".format(results["total_loss"],results["L_p"],results["L_gdl"],results["L_GAN"]))
            elif model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel":
                print ("Total_loss:{}".format(results["total_loss"]))
            elif model.__class__.__name__ == "SAVPVideoPredictionModel":
                print("Total_loss/g_losses:{}; d_losses:{}; g_loss:{}; d_loss: {}".format(results["g_losses"],results["d_losses"],results["g_loss"],results["d_loss"]))
            elif model.__class__.__name__ == "VanillaVAEVideoPredictionModel":
                print("Total_loss:{}; latent_losses:{}; reconst_loss:{}".format(results["total_loss"],results["latent_loss"],results["recon_loss"]))
            else:
                print ("The model name does not exist")

            #print("saving model to", args.output_dir)

            saver.save(sess, os.path.join(args.output_dir, "model"), global_step=step)
            # +++ Scarlet 20200813
            timeit_end = time.time()  
            # --- Scarlet 20200813
            print("time needed for this step", timeit_end - timeit_start, ' s')
            if step % 20 == 0:
                # I save the pickle file and plot here inside the loop in case the training process cannot finished after job is done.
                save_results_to_pkl(train_losses,val_losses,args.output_dir)
                plot_train(train_losses,val_losses,step,args.output_dir)
                                

        train_time = time.time() - run_start_time
        results_dict = {"train_time":train_time,
                        "total_steps":total_steps}
        save_results_to_dict(results_dict,args.output_dir)
        #save_results_to_pkl(train_losses, val_losses, args.output_dir)
        print("train_losses:",train_losses)
        print("val_losses:",val_losses) 
        #plot_train(train_losses,val_losses,args.output_dir)
        print("Done")
        # +++ Scarlet 20200814
        print("Total training time:", train_time/60., "min")
        # +++ Scarlet 20200814
        
if __name__ == '__main__':
    main()
