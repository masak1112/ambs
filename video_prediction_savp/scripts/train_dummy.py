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
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def add_tag_suffix(summary, tag_suffix):
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary)
    summary = summary_proto

    for value in summary.value:
        tag_split = value.tag.split('/')
        value.tag = '/'.join([tag_split[0] + tag_suffix] + tag_split[1:])
    return summary.SerializeToString()

def generate_output_dir(output_dir, model,model_hparams,logs_dir,output_dir_postfix):
    if output_dir is None:
        list_depth = 0
        model_fname = ''
        for t in ('model=%s,%s' % (model, model_hparams)):
            if t == '[':
                list_depth += 1
            if t == ']':
                list_depth -= 1
            if list_depth and t == ',':
                t = '..'
            if t in '=,':
                t = '.'
            if t in '[]':
                t = ''
            model_fname += t
        output_dir = os.path.join(logs_dir, model_fname) + output_dir_postfix
    return output_dir


def get_model_hparams_dict(model_hparams_dict):
    """
    Get model_hparams_dict from json file
    """
    model_hparams_dict_load = {}
    if model_hparams_dict:
        with open(model_hparams_dict) as f:
            model_hparams_dict_load.update(json.loads(f.read()))
    return model_hparams_dict

def resume_checkpoint(resume,checkpoint,output_dir):
    """
    Resume the existing model checkpoints and set checkpoint directory
    """
    if resume:
        if checkpoint:
            raise ValueError('resume and checkpoint cannot both be specified')
        checkpoint = output_dir
    return checkpoint

def set_seed(seed):
    if seed is not None:
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def load_params_from_checkpoints_dir(model_hparams_dict,checkpoint,dataset,model):
   
    model_hparams_dict_load = {}
    if model_hparams_dict:
        with open(model_hparams_dict) as f:
            model_hparams_dict_load.update(json.loads(f.read()))
 
    if checkpoint:
        checkpoint_dir = os.path.normpath(checkpoint)
        if not os.path.isdir(checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            dataset = dataset or options['dataset']
            model = model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict_load.update(json.loads(f.read()))
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
    return dataset, model, model_hparams_dict_load

def setup_dataset(dataset,input_dir,val_input_dir):
    VideoDataset = datasets.get_dataset_class(dataset)
    train_dataset = VideoDataset(
        input_dir,
        mode='train')
    val_dataset = VideoDataset(
        val_input_dir or input_dir,
        mode='val')
    variable_scope = tf.get_variable_scope()
    variable_scope.set_use_resource(True)
    return train_dataset,val_dataset,variable_scope

def setup_model(model,model_hparams_dict,train_dataset,model_hparams):
    """
    Set up model instance
    """
    VideoPredictionModel = models.get_model_class(model)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': train_dataset.hparams.context_frames,
        'sequence_length': train_dataset.hparams.sequence_length,
        'repeat': train_dataset.hparams.time_shift,
    })
    model = VideoPredictionModel(
        hparams_dict=hparams_dict,
        hparams=model_hparams)
    return model

def save_dataset_model_params_to_checkpoint_dir(args,output_dir,train_dataset,model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(train_dataset.hparams.values(), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))
    return None

def make_dataset_iterator(train_dataset, val_dataset, batch_size ):
    train_tf_dataset = train_dataset.make_dataset_v2(batch_size)
    train_iterator = train_tf_dataset.make_one_shot_iterator()
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    train_handle = train_iterator.string_handle()
    val_tf_dataset = val_dataset.make_dataset_v2(batch_size)
    val_iterator = val_tf_dataset.make_one_shot_iterator()
    val_handle = val_iterator.string_handle()
    #iterator = tf.data.Iterator.from_string_handle(
    #    train_handle, train_tf_dataset.output_types, train_tf_dataset.output_shapes)
    inputs = train_iterator.get_next()
    val = val_iterator.get_next()
    return inputs,train_handle, val_handle


def plot_train(train_losses,val_losses,output_dir):
    epochs = list(range(len(train_losses))) 
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir,'plot_train.png'))

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
    parser.add_argument("--val_input_dir", type=str, help="directories containing the tfrecords. default: input_dir")
    parser.add_argument("--logs_dir", default='logs', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams")
    parser.add_argument("--output_dir_postfix", default="")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--resume", action='store_true', help='resume from lastest checkpoint in output_dir.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters") 
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")

    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed",default=1234, type=int)

    args = parser.parse_args()
     
    #Set seed  
    set_seed(args.seed)
    
    #setup output directory
    args.output_dir = generate_output_dir(args.output_dir, args.model, args.model_hparams, args.logs_dir, args.output_dir_postfix)
    
    #resume the existing checkpoint and set up the checkpoint directory to output directory
    args.checkpoint = resume_checkpoint(args.resume,args.checkpoint,args.output_dir)
 
    #get model hparams dict from json file
    #load the existing checkpoint related datasets, model configure (This information was stored in the checkpoint dir when last time training model)
    args.dataset,args.model,model_hparams_dict = load_params_from_checkpoints_dir(args.model_hparams_dict,args.checkpoint,args.dataset,args.model)
     
    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')
    #setup training val datset instance
    train_dataset,val_dataset,variable_scope = setup_dataset(args.dataset,args.input_dir,args.val_input_dir)
    
    #setup model instance 
    model=setup_model(args.model,model_hparams_dict,train_dataset,args.model_hparams)

    batch_size = model.hparams.batch_size
    #Create input and val iterator
    inputs, train_handle, val_handle = make_dataset_iterator(train_dataset, val_dataset, batch_size)
    
    #build model graph
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
    total_steps = steps_per_epoch * max_epochs
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
        start_step = sess.run(model.global_step)
        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        train_losses=[]
        val_losses=[]
        run_start_time = time.time()        
        for step in range(total_steps):
            global_step = sess.run(model.global_step)
            print ("global_step:", global_step)
            val_handle_eval = sess.run(val_handle)
            
            #Fetch variables in the graph
            fetches = {"global_step":model.global_step}
            fetches["train_op"] = model.train_op
            #fetches["latent_loss"] = model.latent_loss
            fetches["total_loss"] = model.total_loss

            #fetch the specific loss function only for mcnet
            if model.__class__.__name__ == "McNetVideoPredictionModel":
                fetches["L_p"] = model.L_p
                fetches["L_gdl"] = model.L_gdl
                fetches["L_GAN"]  =model.L_GAN
            
            fetches["summary"] = model.summary_op       
            results = sess.run(fetches)
            train_losses.append(results["total_loss"])          
            #Fetch losses for validation data
            val_fetches = {}
            #val_fetches["latent_loss"] = model.latent_loss
            val_fetches["total_loss"] = model.total_loss
            val_fetches["summary"] = model.summary_op
            val_results = sess.run(val_fetches,feed_dict={train_handle: val_handle_eval})
            val_losses.append(val_results["total_loss"])
            
            summary_writer.add_summary(results["summary"])
            summary_writer.add_summary(val_results["summary"])
            summary_writer.flush()
             
            # global_step will have the correct step count if we resume from a checkpoint
            # global step is read before it's incemented
            train_epoch = global_step/steps_per_epoch
            print("progress  global step %d  epoch %0.1f" % (global_step + 1, train_epoch))

            if model.__class__.__name__ == "McNetVideoPredictionModel":
              print("Total_loss:{}; L_p_loss:{}; L_gdl:{}; L_GAN: {}".format(results["total_loss"],results["L_p"],results["L_gdl"],results["L_GAN"]))
            elif model.__class__.__name__ == "VanillaConvLstmVideoPredictionModel":
                print ("Total_loss:{}".format(results["total_loss"]))
            else:
                print ("The model name does not exist")
            
            #print("saving model to", args.output_dir)
            saver.save(sess, os.path.join(args.output_dir, "model"), global_step=step)#
        train_time = time.time() - run_start_time
        results_dict = {"train_time":train_time,
                        "total_steps":total_steps}
        save_results_to_dict(results_dict,args.output_dir)
        save_results_to_pkl(train_losses, val_losses, args.output_dir)
        print("train_losses:",train_losses)
        print("val_losses:",val_losses) 
        plot_train(train_losses,val_losses,args.output_dir)
        print("Done")
        
if __name__ == '__main__':
    main()
