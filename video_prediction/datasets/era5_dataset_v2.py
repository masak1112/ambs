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
from DataPreprocess.process_netCDF_v2 import get_unique_vars
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
        self.video_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['sequence_length','height', 'width', 'channels'])
        self.image_shape = self.video_shape[1:]
        self.state_like_names_and_shapes['images'] = 'images/encoded', self.image_shape

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
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'sequence_length': tf.FixedLenFeature([], tf.int64),
                'channels': tf.FixedLenFeature([],tf.int64),
                # 'images/encoded':  tf.FixedLenFeature([], tf.string)
                'images/encoded': tf.VarLenFeature(tf.float32)
            }
            
            # for i in range(20):
            #     keys_to_features["frames/{:04d}".format(i)] = tf.FixedLenFeature((), tf.string)
            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            print ("Parse features", parsed_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
           # width = tf.sparse_tensor_to_dense(parsed_features["width"])
           # height = tf.sparse_tensor_to_dense(parsed_features["height"])
           # channels  = tf.sparse_tensor_to_dense(parsed_features["channels"])
           # sequence_length = tf.sparse_tensor_to_dense(parsed_features["sequence_length"])
            images = []
            # for i in range(20):
            #    images.append(parsed_features["images/encoded"].values[i])
            # images = parsed_features["images/encoded"]
            # images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["images/encoded"].values[i]),offsets)
            # seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"], '')
            # Parse the string into an array of pixels corresponding to the image
            # images = tf.decode_raw(parsed_features["images/encoded"],tf.int32)

            # images = seq
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
            
class norm_data:
    
    ### set known norms and the requested statistics (to be retrieved from statistics.json) here ###
    known_norms = {}
    known_norms["minmax"] = ["min","max"]
    known_norms["znorm"]  = ["avg","sigma"]
    
    def __init__(self,varnames):
        varnames_uni, _, nvars = get_unique_vars(varnames)
        
        self.varnames = varnames_uni
        self.status_ok= False
            
    def check_and_set_norm(self,stat_dict,norm):
        
        if not norm in self.known_norms.keys():
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '"+norm+"' is unknown.")
       
        if not all(items in stat_dict for items in self.varnames):
            print("Keys in stat_dict:")
            print(stat_dict.keys())
            
            print("Requested variables:")
            print(self.varnames)
            raise ValueError("Could not find all requested variables in statistics dictionary.")   

        for varname in self.varnames:
            for stat_name in self.known_norms[norm]:
                setattr(self,varname+stat_name,stat_dict[varname][0][stat_name])
                
        self.status_ok = True
                
    def norm_var(self,data,varname,norm):
        
        # some sanity checks
        if not self.status_ok: raise ValueError("norm_data-object needs to be initialized and checked first.")
        
        if not norm in self.known_norms.keys():
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '"+norm+"' is unknown.")
        
        
        if norm == "minmax":
            return((data[...] - getattr(self,varname+"min"))/(getattr(self,varname+"max") - getattr(self,varname+"min")))
        elif norm == "znorm":
            return((data[...] - getattr(self,varname+"avg"))/getattr(self,varname+"sigma")**2)
        
    def denorm_var(self,data,varname,norm):
        
        # some sanity checks
        if not self.status_ok: raise ValueError("norm_data-object needs to be initialized and checked first.")        
        
        if not norm in self.known_norms.keys():
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '"+norm+"' is unknown.")
        
        if norm == "minmax":
            return(data[...] * (getattr(self,varname+"max") - getattr(self,varname+"min")) + getattr(self,varname+"max"))
        elif norm == "znorm":
            return(data[...] * getattr(self,varname+"sigma")**2 + getattr(self,varname+"avg"))
        

def read_frames_and_save_tf_records(output_dir,input_dir,partition_name,vars_in,seq_length=20,sequences_per_file=128,height=64,width=64,channels=3,**kwargs):#Bing: original 128
    # ML 2020/04/08:
    # Include vars_in for more flexible data handling (normalization and reshaping)
    # and optional keyword argument for kind of normalization
    
    if 'norm' in kwargs:
        norm = kwargs.get("norm")
    else:
        norm = "minmax"
        print("Make use of default minmax-normalization...")

    output_dir = os.path.join(output_dir,partition_name)
    os.makedirs(output_dir,exist_ok=True)
    
    norm_cls  = norm_data(vars_in)
    nvars     = len(vars_in)
    
    # open statistics file and store the dictionary
    with open(os.path.join(input_dir,"statistics.json")) as js_file:
        norm_cls.check_and_set_norm(json.load(js_file),norm)        
    
    sequences = []
    sequence_iter = 0
    sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
    X_train = hkl.load(os.path.join(input_dir, "X_" + partition_name + ".hkl"))
    X_possible_starts = [i for i in range(len(X_train) - seq_length)]
    for X_start in X_possible_starts:
        print("Interation", sequence_iter)
        X_end = X_start + seq_length
        #seq = X_train[X_start:X_end, :, :,:]
        seq = X_train[X_start:X_end,:,:]
        #print("*****len of seq ***.{}".format(len(seq)))
        #seq = list(np.array(seq).reshape((len(seq), 64, 64, 3)))
        seq = list(np.array(seq).reshape((seq_length, height, width, nvars)))
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
            for i in range(nvars):    
                sequences[:,:,:,:,i] = norm_cls.norm_var(sequences[:,:,:,:,i],vars_in[i],norm)

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
    parser.add_argument("-height",type=int,default=64)
    parser.add_argument("-width",type = int,default=64)
    parser.add_argument("-seq_length",type=int,default=20)
    args = parser.parse_args()
    current_path = os.getcwd()
    #input_dir = "/Users/gongbing/PycharmProjects/video_prediction/splits"
    #output_dir = "/Users/gongbing/PycharmProjects/video_prediction/data/era5"
    partition_names = ['train','val',  'test'] #64,64,3 val has issue#
  
    for partition_name in partition_names:
        read_frames_and_save_tf_records(output_dir=args.output_dir,input_dir=args.input_dir,vars_in=args.variables,partition_name=partition_name, seq_length=args.seq_length,height=args.height,width=args.width,sequences_per_file=2) #Bing: Todo need check the N_seq
        #ead_frames_and_save_tf_records(output_dir = output_dir, input_dir = input_dir,partition_name = partition_name, N_seq=20) #Bing: TODO: first try for N_seq is 10, but it met loading data issue. let's try 5

if __name__ == '__main__':
    main()

