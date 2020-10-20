
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"

import argparse
import os
import glob
import sys
import itertools
import pickle
import random
import re
import numpy as np
import json
import tensorflow as tf
from collections import OrderedDict
from tensorflow.contrib.training import HParams
from mpi4py import MPI
from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
import data_preprocess.process_netCDF_v2
from general_utils import get_unique_vars
from statistics import Calc_data_stat
from metadata import MetaData
from normalization import Norm_data
from google.protobuf.json_format import MessageToDict
import datetime


class ERA5Pkl2Tfrecords(object):
    def __init__(self,input_dir=None, output_dir=None,datasplit_config=None,vars_in=None,hparams_dict=None,sequences_per_file=128,norm="minmax"):
        """
        This class is used for convert pkl files to tfrecords
        args:
            input_dir          : str, the parament path of pkl files directiory. This directory should be at "year" level
            outpout_dir        : str, the path to save the tfrecords files 
            datasplit_config   : the path pointing to the datasplit_config jason file
            hparams_dict_path  : a dict that contains hparameters,
            sequences_per_file : int, how many sequences/samples per tfrecord to be saved
            norm               :str, normalization methods from Norm_data class, "minmax" or "znorm" default is "minmax", 
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        #if the output_dir is not exist, then create it
        os.makedirs(self.output_dir,exist_ok=True)
        #get metadata,includes the var_in, image height, width etc.
        self.get_metadata()
        #Get the data split informaiton
        self.datasplit_dict_path = datasplit_config
        self.data_dict = self.get_datasplit()
        self.hparams_dict = hparams_dict
        self.hparams = self.parse_hparams()
        self.sequence_length = self.hparams.sequence_length
        if norm == "minmax" or norm == "znorm":
            self.norm = norm
        else:
            raise ("norm should be either 'minmax' or 'znorm'") 
        self.sequences_per_file = sequences_per_file
        


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
            loss_fun = "rmse"
        )
        return hparams

    def get_datasplit(self):
        """
        Get the datasplit json file
        """

        with open(self.datasplit_dict_path) as f:
            self.d = json.load(f)
        return self.d

    def parse_hparams(self):
        """
        Parse the hparams setting to ovoerride the default ones
        """
        parsed_hparams = self.get_default_hparams().override_from_dict(self.hparams_dict or {})
        return parsed_hparams


    def get_months(self):
        """
        Get the months in the datasplit_config
        Return : a 1-dim array contains the months set
        """
        self.mode_list = []
        self.years = []
        self.months = []
        for mode, value in self.d.items():
            self.mode_list.append(mode)
            for year, month in value.items():
                self.years.append(year)
                self.months.extend(month)
        return set(self.months)

    def get_stats_file(self):
        """
        Get the correspoding statistic file
        """
        pkl_dir = os.path.join(self.input_dir,"pickle")
        print("Opening json-file: " + os.path.join(pkl_dir,"statistics.json"))
        self.stats_file = os.path.join(pkl_dir,"statistics.json")
        if os.path.isfile(self.stats_file):
            with open(self.stats_file) as js_file:
                self.stats = json.load(js_file)
        else:
            raise ("statistic file does not exist")

    def get_metadata(self):
        """
        This function get the meta data that generared from data_process_step1, we aim to extract the height and width informaion from it
        vars_in: list(str), must be consistent with the list from DataPreprocessing_step1
        height: int, the height of the image
        width: int the width of the image
        """
        metadata_fl = os.path.join(self.input_dir,"metadata.json")
        if os.path.isfile(metadata_fl):
            self.metadata_fl = metadata_fl
            with open(self.metadata_fl) as f:
                self.metadata = json.load(f)
            self.frame_size = self.metadata["frame_size"]
            self.height = self.frame_size["nx"]
            self.width = self.frame_size["ny"]
            self.variables = self.metadata["variables"]
            self.vars_in = [list(var.values())[0] for var in self.variables]
           
        else:
            raise ("The metadata_file is not generated properly, you might need to re-run previous step of the workflow")


    @staticmethod
    def save_tf_record(output_fname, sequences, t_start_points):
        """
        Save the squences, and the corresdponding timestamp start point to tfrecords
        args:
            output_frames:str, the file names of the output
            sequences: list or array, the sequences want to be saved to tfrecords, [sequences,seq_len,height,width,channels]
            t_start_points: datetime type in the list,  the first timestamp for each sequence [seq_len,height,width, channel], the len of t_start_points is the same as sequences
        """
        sequences = np.array(sequences)

        assert sequences.shape[0] == len(t_start_points)
        assert type(t_start_points[0]) == datetime.datetime

        with tf.python_io.TFRecordWriter(output_fname) as writer:
            for i in range(len(sequences)):
                sequence = sequences[i]

                t_start = t_start_points[i].strftime("%Y%m%d%H")
                num_frames = len(sequence)
                height, width, channels = sequence[0].shape
                encoded_sequence = np.array([list(image) for image in sequence])
                features = tf.train.Features(feature={
                    'sequence_length': _int64_feature(num_frames),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'channels': _int64_feature(channels),
                    't_start': _int64_feature(int(t_start)),
                    'images/encoded': _floats_feature(encoded_sequence.flatten()),
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())



    def initia_norm_class(self):
        """
        Get normalization data class 
        """
        print("Make use of default minmax-normalization...")
        # init normalization-instance
        self.norm_cls = Norm_data(self.vars_in)
        self.nvars = len(self.vars_in)
        #get statistic file
        self.get_stats_file()
        # open statistics file and feed it to norm-instance
        self.norm_cls.check_and_set_norm(self.stats,self.norm)


    def normalize_vars_per_seq(self,sequences):
        """
        Normalize all the variables for the sequences
        args:
            sequences: list or array, is the sequences need to be saved to tfrecorcd. the shape should be [sequences_per_file,seq_length,height,width,nvars]
        Return:
            the normalized sequences
        """
        assert len(np.array(sequences).shape) == 5
        #Normalization should adpot the selected variables, here we used duplicated channel temperature variables
        sequences = np.array(sequences)
        #normalization
        for i in range(self.nvars):
            sequences[:,:,:,:,i] = self.norm_cls.norm_var(sequences[:,:,:,:,i],self.vars_in[i],self.norm)
        return sequences

    
    def read_pkl_and_save_tfrecords(self,year,month):
        """
        Read pickle files based on month, to process and save to tfrecords,
        args:
            year: int, the target year to save to tfrecord
            month: int, the target month to save to tfrecord 
        """
        #Define the input_file based on the year and month
        self.input_file_year = os.path.join(os.path.join(self.input_dir, "pickle"),str(year))
        input_file = os.path.join(self.input_file_year,'X_{:02d}.pkl'.format(month))
        temp_input_file = os.path.join(self.input_file_year,'T_{:02d}.pkl'.format(month))

        self.initia_norm_class()
        sequences = []
        t_start_points = []
        sequence_iter = 0

        #try:
        with open(input_file, "rb") as data_file:
            X_train = pickle.load(data_file)
        with open(temp_input_file,"rb") as temp_file:
             T_train = pickle.load(temp_file)

        #check to make sure the X_train and T_train has the same length 
        assert (len(X_train) == len(T_train))

        X_possible_starts = [i for i in range(len(X_train) - self.sequence_length)]
        for X_start in X_possible_starts:
            X_end = X_start + self.sequence_length
            seq = X_train[X_start:X_end,:,:,:]
            #Recored the start point of the timestamps
            t_start = T_train[X_start]  
            seq = list(np.array(seq).reshape((self.sequence_length, self.height, self.width, self.nvars)))
            if not sequences:
                last_start_sequence_iter = sequence_iter
            sequences.append(seq)
            t_start_points.append(t_start[0])
            sequence_iter += 1

            if len(sequences) == self.sequences_per_file:
                #Nomalize variables in the sequence
                sequences = ERA5Pkl2Tfrecords.normalize_vars_per_seq(self,sequences)
                output_fname = 'sequence_Y_{}_M_{}_{}_to_{}.tfrecords'.format(year,month,last_start_sequence_iter,sequence_iter - 1)
                output_fname = os.path.join(self.output_dir, output_fname)
                #Write to tfrecord
                ERA5Pkl2Tfrecords.write_seq_to_tfrecord(output_fname,sequences,t_start_points)
                t_start_points = []
                sequences = []
            print("Finished for input file",input_file)

#         except FileNotFoundError as fnf_error:
#             print(fnf_error)


    @staticmethod
    def write_seq_to_tfrecord(output_fname,sequences,t_start_points):
        """
        Function to check if the sequences has been processed. if yes, will skip it, otherwise save the sequences to output file
        """
        if os.path.isfile(output_fname):
            print(output_fname, 'already exists, skip it')
        else:
            ERA5Pkl2Tfrecords.save_tf_record(output_fname, list(sequences), t_start_points)   



class ERA5Dataset(ERA5Pkl2Tfrecords):

    def __init__(self, mode='train', seed=None, **kwargs):
        """
        This class is used for preparing data for training/validation and test models
        args:
            mode: string, "train","val" or "test"
            seed: int, the seed for dataset 
        """
        super(ERA5Dataset, self).__init__(**kwargs)
        self.input_dir_tfrecords = os.path.join(self.input_dir,"tfrecords")
        self.mode = mode
        self.seed = seed
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)
        if not os.path.exists(self.input_dir_tfrecords):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir_tfrecords)
        self.max_epochs = self.hparams.max_epochs


    def get_tfrecords_filesnames_base_datasplit(self):
        """
        Get  absolute .tfrecord path names based on the data splits patterns
        """
        self.filenames = []
        self.data_mode = self.data_dict[self.mode]
        self.tf_names = []
        print("data_mode:",self.data_mode)
        for year, months in self.data_mode.items():
            for month in months:
                tf_files = "sequence_Y_{}_M_{}_*_to_*.tfrecord*".format(year,month)    
                self.tf_names.append(tf_files)
        # look for tfrecords in input_dir and input_dir/mode directories
        for files in self.tf_names:
            self.filenames.extend(glob.glob(os.path.join(self.input_dir_tfrecords, files)))
        if self.filenames:
            self.filenames = sorted(self.filenames)  # ensures order is the same across systems
        if not self.filenames:
            raise FileNotFoundError('No tfrecords were found in %s.' % self.input_dir_tfrecords)


    def get_example_info(self):
        """
        Get the data information from tfrecord file
        """
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        print("features in dataset:",feature.keys())
        self.video_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['sequence_length','height', 'width', 'channels'])
        self.image_shape = self.video_shape[1:]
 
    def calculate_samples(self):
        pass

#     def write_sequence_file(output_dir,seq_length,sequences_per_file):
#         partition_names = ["train","val","test"]
#         for partition_name in partition_names:
#             save_output_dir = os.path.join(output_dir,partition_name)
#             tfCounter = len(glob.glob1(save_output_dir,"*.tfrecords"))
#             print("Partition_name: {}, number of tfrecords: {}".format(partition_name,tfCounter))
#             sequence_lengths_file = open(os.path.join(save_output_dir, 'sequence_lengths.txt'), 'w')
#             for i in range(tfCounter*sequences_per_file):
#                 sequence_lengths_file.write("%d\n" % seq_length)
#             sequence_lengths_file.close()
            
            
#     @property
#     def jpeg_encoding(self):
#         return False


#     def num_examples_per_epoch(self):
#         with open(os.path.join(self.input_dir, 'sequence_lengths.txt'), 'r') as sequence_lengths_file:
#             sequence_lengths = sequence_lengths_file.readlines()
#         sequence_lengths = [int(sequence_length.strip()) for sequence_length in sequence_lengths]
#         return np.sum(np.array(sequence_lengths) >= self.hparams.sequence_length)

#     def filter(self, serialized_example):
#         return tf.convert_to_tensor(True)

#     def make_dataset(self, batch_size):
#         def parser(serialized_example):
#             seqs = OrderedDict()
#             keys_to_features = {
#                 'width': tf.FixedLenFeature([], tf.int64),
#                 'height': tf.FixedLenFeature([], tf.int64),
#                 'sequence_length': tf.FixedLenFeature([], tf.int64),
#                 'channels': tf.FixedLenFeature([],tf.int64),
#                 't_start':  tf.VarLenFeature(tf.int64),
#                 'images/encoded': tf.VarLenFeature(tf.float32)
#             }

#             parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
#             seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
#             T_start = tf.sparse_tensor_to_dense(parsed_features["t_start"])
#             images = []
#             print("Image shape {}, {},{},{}".format(self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]))
#             images = tf.reshape(seq, [self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]], name = "reshape_new")
#             seqs["images"] = images
#             seqs["T_start"] = T_start
#             return seqs
#         filenames = self.filenames
#         shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
#         if shuffle:
#             random.shuffle(filenames)
#         dataset = tf.data.TFRecordDataset(filenames, buffer_size = 8* 1024 * 1024) 
#         dataset = dataset.filter(self.filter)
#         if shuffle:
#             dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size =1024, count = self.num_epochs))
#         else:
#             dataset = dataset.repeat(self.num_epochs)
#         num_parallel_calls = None if shuffle else 1
#         dataset = dataset.apply(tf.contrib.data.map_and_batch(
#             parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))s
#         dataset = dataset.prefetch(batch_size)  
#         return dataset
    
#     def make_batch(self, batch_size):
#         dataset = self.make_dataset_v2(batch_size)
#         iterator = dataset.make_one_shot_iterator()
#         return iterator.get_next()
    
    
    
    
    
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_model_hparams_dict(model_hparams_dict_path):
    """
    Get model_hparams_dict from json file
    """
    model_hparams_dict_load = {}
    if model_hparams_dict_path:
        with open(model_hparams_dict_path) as f:
            model_hparams_dict_load.update(json.loads(f.read()))
    return model_hparams_dict_load





if __name__ == "__main__":

    input_dir =  "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2017to2017M01to12_wb025-160x128-2970N1500W-T2_MSL_gph500_test/"
    output_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test/tfrecords/"
    datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/cv_test.json"
    hparams_path = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams.json"
    model_hparams_dict = get_model_hparams_dict(hparams_path)
    case1 = ERA5Dataset(seed=1234,input_dir=input_dir,output_dir=output_dir,datasplit_config=datasplit_config,hparams_dict=model_hparams_dict,sequences_per_file=128,vars_in=["T2","MSL","gph500"])
    case1.get_tfrecords_filesnames_base_datasplit()
