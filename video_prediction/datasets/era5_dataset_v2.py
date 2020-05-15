import argparse
import glob
import itertools
import os
import pickle
import random
import re
import hickle as hkl
import numpy as np
import json
import tensorflow as tf
from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
# ML 2020/04/14: hack for getting functions of process_netCDF_v2:
from os import path
import sys
sys.path.append(path.abspath('../../workflow_parallel_frame_prediction/'))
from DataPreprocess.process_netCDF_v2 import get_stat
from DataPreprocess.process_netCDF_v2 import get_stat_allvars
#from base_dataset import VarLenFeatureVideoDataset
from collections import OrderedDict
from tensorflow.contrib.training import HParams

class ERA5Dataset_v2(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(ERA5Dataset_v2, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['height', 'width', 'channels'])
        self.state_like_names_and_shapes['images'] = 'images/encoded', image_shape

    def get_default_hparams_dict(self):
        default_hparams = super(ERA5Dataset_v2, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,#Bing: Todo oriignal is 10
            sequence_length=20,#bing: TODO original is 20,
            long_sequence_length=20,
            force_time_shift=True,
            shuffle_on_val=True, 
            use_state=False,
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
                # 'width': tf.FixedLenFeature([], tf.int64),
                # 'height': tf.FixedLenFeature([], tf.int64),
                'sequence_length': tf.FixedLenFeature([], tf.int64),
                # 'channels': tf.FixedLenFeature([],tf.int64),
                # 'images/encoded':  tf.FixedLenFeature([], tf.string)
                'images/encoded': tf.VarLenFeature(tf.float32)
            }
            # for i in range(20):
            #     keys_to_features["frames/{:04d}".format(i)] = tf.FixedLenFeature((), tf.string)
            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
            images = []
            # for i in range(20):
            #    images.append(parsed_features["images/encoded"].values[i])
            # images = parsed_features["images/encoded"]
            # images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["images/encoded"].values[i]),offsets)
            # seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"], '')
            # Parse the string into an array of pixels corresponding to the image
            # images = tf.decode_raw(parsed_features["images/encoded"],tf.int32)

            # images = seq
            images = tf.reshape(seq, [20, 128, 160, 3], name = "reshape_new")
            print("IMAGES", images)
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
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence in sequences:
            num_frames = len(sequence)
            height, width, channels = sequence[0].shape
            encoded_sequence = np.array([list(image) for image in sequence])

            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'images/encoded': _floats_feature(encoded_sequence.flatten()),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

def read_frames_and_save_tf_records(output_dir,input_dir,partition_name,vars_in,N_seq,sequences_per_file=128,**kwargs):#Bing: original 128
    # ML 2020/04/08:
    # Include vars_in for more flexible data handling (normalization and reshaping)
    # and optional keyword argument for kind of normalization
    known_norms = ["minmax"]     # may be more elegant to define a class here?   

    output_dir = os.path.join(output_dir,partition_name)
    os.makedirs(output_dir,exist_ok=True)
    
    nvars     = len(vars_in)
    vars_uni, indrev = np.unique(vars_in,return_inverse=True)
    if 'norm' in kwargs:
        norm = kwargs.get("norm")
        if (not norm in knwon_norms): 
            raise ValueError("Pass valid normalization identifier.")
            print("Known identifiers are: ")
            for norm_name in known_norm:
                print('"'+norm_name+'"')
    else:
        norm = "minmax"
    
    # open statistics file
    with open(os.path.join(input_dir,"statistics.json")) as js_file:
        data = json.load(js_file)
    
        if (norm == "minmax"):
            varmin, varmax = get_stat_allvars(data,"min",vars_in), get_stat_allvars(data,"max",vars_in)

    #print(len(varmin))
    #print(varmin)
    
    sequences = []
    sequence_iter = 0
    sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
    X_train = hkl.load(os.path.join(input_dir, "X_" + partition_name + ".hkl"))
    X_possible_starts = [i for i in range(len(X_train) - N_seq)]
    for X_start in X_possible_starts:
        print("Interation", sequence_iter)
        X_end = X_start + N_seq
        #seq = X_train[X_start:X_end, :, :,:]
        seq = X_train[X_start:X_end,:,:]
        #print("*****len of seq ***.{}".format(len(seq)))
        #seq = list(np.array(seq).reshape((len(seq), 64, 64, 3)))
        seq = list(np.array(seq).reshape((len(seq), 128, 160,nvars)))
        if not sequences:
            last_start_sequence_iter = sequence_iter
            print("reading sequences starting at sequence %d" % sequence_iter)
        sequences.append(seq)
        sequence_iter += 1
        sequence_lengths_file.write("%d\n" % len(seq))

        if len(sequences) == sequences_per_file:
            ###Normalization should adpot the selected variables, here we used duplicated channel temperature variables
            sequences = np.array(sequences)
            ### normalization
            # ML 2020/04/08:
            # again rather inelegant/inefficient as...
            # a) normalization should be cast in class definition (with initialization, setting of norm. approach including 
            #    data retrieval and the normalization itself
            for i in range(nvars):    
                sequences[:,:,:,:,i] = (sequences[:,:,:,:,i]-varmin[i])/(varmax[i]-varmin[i])

            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter - 1)
            output_fname = os.path.join(output_dir, output_fname)
            save_tf_record(output_fname, list(sequences))
            sequences = []
    sequence_lengths_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories ""boxing, handclapping, handwaving, ""jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    # ML 2020/04/08 S
    # Add vars for ensuring proper normalization and reshaping of sequences
    parser.add_argument("-vars","--variables",dest="variables", nargs='+', type=str, help="Names of input variables.")
    # parser.add_argument("image_size_h", type=int)
    # parser.add_argument("image_size_v", type = int)
    args = parser.parse_args()
    current_path = os.getcwd()
    #input_dir = "/Users/gongbing/PycharmProjects/video_prediction/splits"
    #output_dir = "/Users/gongbing/PycharmProjects/video_prediction/data/era5"
    partition_names = ['train','val',  'test'] #64,64,3 val has issue#
  
    for partition_name in partition_names:
        read_frames_and_save_tf_records(output_dir=args.output_dir,input_dir=args.input_dir,vars_in=args.variables,partition_name=partition_name, N_seq=20, sequences_per_file=2) #Bing: Todo need check the N_seq
        #ead_frames_and_save_tf_records(output_dir = output_dir, input_dir = input_dir,partition_name = partition_name, N_seq=20) #Bing: TODO: first try for N_seq is 10, but it met loading data issue. let's try 5

if __name__ == '__main__':
    main()

