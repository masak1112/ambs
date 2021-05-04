import argparse
import sys
import glob
import itertools
import os
import pickle
import random
import re
import numpy as np
import json
import tensorflow as tf
from tensorflow.contrib.training import HParams
from mpi4py import MPI
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model_modules.video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
import data_preprocess.process_netCDF_v2 
from general_utils import get_unique_vars
from statistics import Calc_data_stat 
from metadata import MetaData

class MovingMnist(object):
    def __init__(self, input_dir=None, datasplit_config=None, hparams_dict_config=None, mode="train",seed=None):
        """
        This class is used for preparing the data for moving mnist, and split the data to train/val/testing
        :params input_dir: the path of tfrecords files 
        :params datasplit_config: the path pointing to the datasplit_config json file
        :params hparams_dict_config: the path to the dict that contains hparameters
        :params mode: string, "train","val" or "test"
        :params seed:int, the seed for dataset 
        :return None
        """
        self.input_dir = input_dir
        self.mode = mode 
        self.seed = seed
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir)
        self.datasplit_dict_path = datasplit_config
        self.data_dict = self.get_datasplit()
        self.hparams_dict_config = hparams_dict_config
        self.hparams_dict = self.get_model_hparams_dict()
        self.hparams = self.parse_hparams() 


    def get_datasplit(self):
        """
        Get the datasplit json file
        """
        with open(self.datasplit_dict_path) as f:
            self.d = json.load(f)
        return self.d



    def get_model_hparams_dict(self):
        """
        Get model_hparams_dict from json file
        """
        self.model_hparams_dict_load = {}
        if self.hparams_dict_config:
            with open(self.hparams_dict_config) as f:
                self.model_hparams_dict_load.update(json.loads(f.read()))
        return self.model_hparams_dict_load

                     
    def parse_hparams(self):
        """
        Parse the hparams setting to ovoerride the default ones
        """
        parsed_hparams = self.get_default_hparams().override_from_dict(self.hparams_dict or {})
        return parsed_hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())


    def get_default_hparams_dict(self):
        """
        The function that contains default hparams
        Returns:
            A dict with the following hyperparameters.
            context_frames  : the number of ground-truth frames to pass in at start.
            sequence_length : the number of frames in the video sequence 
            max_epochs      : the number of epochs to train model
            lr              : learning rate
            loss_fun        : the loss function
        """
        hparams = dict(
            context_frames=10,
            sequence_length=20,
            max_epochs = 20,
            batch_size = 40,
            lr = 0.001,
            loss_fun = "rmse",
            shuffle_on_val= True,
        )
        return hparams





   def get_tfrecords_filename_base_datasplit(self):
       """
       Get obsoluate .tfrecords names based on the data splits patterns
       """
       self.filenames = []
       self.data_mode = self.data_dict[self.mode]
       self.tf_names = []
       for indice_group, index in self.data_mode.items():
           for indice in index:
                     















    def num_examples_per_epoch(self):
        with open(os.path.join(self.input_dir, 'number_squences.txt'), 'r') as sequence_lengths_file:
            sequence_lengths = sequence_lengths_file.readlines()
        sequence_lengths = [int(sequence_length.strip()) for sequence_length in sequence_lengths]
        return np.sum(np.array(sequence_lengths) >= self.hparams.sequence_length)

    def filter(self, serialized_example):
        return tf.convert_to_tensor(True)

    def make_dataset_v2(self, batch_size):
        def parser(serialized_example):
            seqs = OrderedDict()
            keys_to_features = {
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'sequence_length': tf.FixedLenFeature([], tf.int64),
                'channels': tf.FixedLenFeature([], tf.int64),
                'images/encoded': tf.VarLenFeature(tf.float32)
            }
            
            # for i in range(20):
            #     keys_to_features["frames/{:04d}".format(i)] = tf.FixedLenFeature((), tf.string)
            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            print ("Parse features", parsed_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
            #width = tf.sparse_tensor_to_dense(parsed_features["width"])
           # height = tf.sparse_tensor_to_dense(parsed_features["height"])
           # channels  = tf.sparse_tensor_to_dense(parsed_features["channels"])
           # sequence_length = tf.sparse_tensor_to_dense(parsed_features["sequence_length"])
            images = []
            print("Image shape {}, {},{},{}".format(self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]))
            images = tf.reshape(seq, [self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]], name = "reshape_new")
            seqs["images"] = images
            return seqs
        filenames = self.filenames
      
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        if shuffle:
            random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames, buffer_size = 8* 1024 * 1024)  # todo: what is buffer_size
        print("files", self.filenames)
        print("mode", self.mode)
        dataset = dataset.filter(self.filter)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size =1024, count = self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)
        
        num_parallel_calls = None if shuffle else 1
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        #dataset = dataset.map(parser)
        # num_parallel_calls = None if shuffle else 1  # for reproducibility (e.g. sampled subclips from the test set)
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #    _parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls)) #  Bing: Parallel data mapping, num_parallel_calls normally depends on the hardware, however, normally should be equal to be the usalbe number of CPUs
        dataset = dataset.prefetch(batch_size)  # Bing: Take the data to buffer inorder to save the waiting time for GPU
        return dataset


    def make_batch(self, batch_size):
        dataset = self.make_dataset_v2(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def save_tf_record(output_fname, sequences):
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for i in range(len(sequences)):
            sequence = sequences[:,i,:,:,:] 
            num_frames = len(sequence)
            height, width = sequence[0,:,:,0].shape
            encoded_sequence = np.array([list(image) for image in sequence])
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(1),
                'images/encoded': _floats_feature(encoded_sequence.flatten()),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

def read_frames_and_save_tf_records(output_dir,dat_npz, seq_length=20, sequences_per_file=128, height=64, width=64):#Bing: original 128
    """
    Read the moving_mnst data which is npz format, and save it to tfrecords files
    The shape of dat_npz is [seq_length,number_samples,height,width]
    moving_mnst only has one channel

    """
    os.makedirs(output_dir,exist_ok=True)
    idx = 0
    num_samples = dat_npz.shape[1]
    dat_npz = np.expand_dims(dat_npz, axis=4) #add one dim to represent channel, then got [seq_length,num_samples,height,width,channel]
    print("data_npz_shape",dat_npz.shape)
    dat_npz = dat_npz.astype(np.float32)
    dat_npz /= 255.0 #normalize RGB codes by dividing it to the max RGB value 
    while idx < num_samples - sequences_per_file:
        sequences = dat_npz[:,idx:idx+sequences_per_file,:,:,:]
        output_fname = 'sequence_{}_{}.tfrecords'.format(idx,idx+sequences_per_file)
        output_fname = os.path.join(output_dir, output_fname)
        save_tf_record(output_fname, sequences)
        idx = idx + sequences_per_file
    return None


def write_sequence_file(output_dir,seq_length,sequences_per_file):    
    partition_names = ["train","val","test"]
    for partition_name in partition_names:
        save_output_dir = os.path.join(output_dir,partition_name)
        tfCounter = len(glob.glob1(save_output_dir,"*.tfrecords"))
        print("Partition_name: {}, number of tfrecords: {}".format(partition_name,tfCounter))
        sequence_lengths_file = open(os.path.join(save_output_dir, 'sequence_lengths.txt'), 'w')
        for i in range(tfCounter*sequences_per_file):
            sequence_lengths_file.write("%d\n" % seq_length)
        sequence_lengths_file.close()


def plot_seq_imgs(imgs,output_png_dir,idx,label="Ground Truth"):
    """
    Plot the seq images 
    """

    if len(np.array(imgs).shape)!=3:raise("img dims should be three: (seq_len,lat,lon)")
    img_len = imgs.shape[0]
    fig = plt.figure(figsize=(18,6))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace = 0., hspace = 0.)
    for i in range(img_len):
        ax1 = plt.subplot(gs[i])
        plt.imshow(imgs[i] ,cmap = 'jet')
        plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])
    plt.savefig(os.path.join(output_png_dir, label + "_" +   str(idx) +  ".jpg"))
    print("images_saved")
    plt.clf()


    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories ""boxing, handclapping, handwaving, ""jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-sequences_per_file",type=int,default=2)
    args = parser.parse_args()
    current_path = os.getcwd()
    data = np.load(os.path.join(args.input_dir,"mnist_test_seq.npy"))
    print("data in minist_test_Seq shape",data.shape)
    seq_length =  data.shape[0]
    height = data.shape[2]
    width = data.shape[3]
    num_samples = data.shape[1] 
    max_npz = np.max(data)
    min_npz = np.min(data)
    print("max_npz,",max_npz)
    print("min_npz",min_npz)
    #Todo need to discuss how to split the data, since we have totally 10000 samples, the origin paper convLSTM used 10000 as training, 2000 as validation and 3000 for testing
    dat_train = data[:,:6000,:,:]
    dat_val = data[:,6000:7000,:,:]
    dat_test = data[:,7000:,:]
    #plot_seq_imgs(dat_test[10:,0,:,:],output_png_dir="/p/project/deepacf/deeprain/video_prediction_shared_folder/results/moving_mnist/convLSTM",idx=1,label="Ground Truth from npz")
    #save train
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"train"),dat_train, seq_length=20, sequences_per_file=40, height=height, width=width)
    #save val
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"val"),dat_val, seq_length=20, sequences_per_file=40, height=height, width=width)
    #save test     
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"test"),dat_test, seq_length=20, sequences_per_file=40, height=height, width=width)
    #write_sequence_file(output_dir=args.output_dir,seq_length=20,sequences_per_file=40)
if __name__ == '__main__':
     main()

