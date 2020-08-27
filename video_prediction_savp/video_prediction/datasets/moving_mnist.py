import argparse
import glob
import itertools
import os
import pickle
import random
import re
import numpy as np
import json
import tensorflow as tf
from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
# ML 2020/04/14: hack for getting functions of process_netCDF_v2:
from os import path
import sys
sys.path.append(path.abspath('../../workflow_parallel_frame_prediction/'))
import DataPreprocess.process_netCDF_v2 
from DataPreprocess.process_netCDF_v2 import get_unique_vars
from DataPreprocess.process_netCDF_v2 import Calc_data_stat
from metadata import MetaData
#from base_dataset import VarLenFeatureVideoDataset
from collections import OrderedDict
from tensorflow.contrib.training import HParams
from mpi4py import MPI
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MovingMnist(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(MovingMnist, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        print("features in dataset:",feature.keys())
        self.video_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['sequence_length','height', 'width', 'channels'])
        self.image_shape = self.video_shape[1:]
        self.state_like_names_and_shapes['images'] = 'images/encoded', self.image_shape

    def get_default_hparams_dict(self):
        default_hparams = super(MovingMnist, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,#Bing: Todo oriignal is 10
            sequence_length=20,#bing: TODO original is 20,
            shuffle_on_val=True, 
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))


    @property
    def jpeg_encoding(self):
        return False



    def num_examples_per_epoch(self):
        with open(os.path.join(self.input_dir, 'sequence_lengths.txt'), 'r') as sequence_lengths_file:
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
        print ("FILENAMES",filenames)
	    #TODO:
	    #temporal_filenames = self.temporal_filenames
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
    
    #Todo need to discuss how to split the data, since we have totally 10000 samples, the origin paper convLSTM used 10000 as training, 2000 as validation and 3000 for testing
    dat_train = data[:,:6000,:,:]
    dat_val = data[:,6000:7000,:,:]
    dat_test = data[:,7000:,:]
    plot_seq_imgs(dat_test[10:,0,:,:],output_png_dir="/p/project/deepacf/deeprain/video_prediction_shared_folder/results/moving_mnist/convLSTM",idx=1,label="Ground Truth from npz")
    #save train
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"train"),dat_train, seq_length=20, sequences_per_file=40, height=height, width=width)
    #save val
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"val"),dat_val, seq_length=20, sequences_per_file=40, height=height, width=width)
    #save test     
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"test"),dat_test, seq_length=20, sequences_per_file=40, height=height, width=width)
    #write_sequence_file(output_dir=args.output_dir,seq_length=20,sequences_per_file=40)
if __name__ == '__main__':
     main()

