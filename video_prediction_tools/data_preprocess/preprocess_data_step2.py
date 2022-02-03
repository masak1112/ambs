# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Class and functions required for preprocessing ERA5 data (preprocessing substep 2)
"""
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020_12_29"


# import modules
import os
import glob
import pickle
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from normalization import Norm_data
from metadata import MetaData
import datetime
from model_modules.video_prediction.datasets import ERA5Dataset


class ERA5Pkl2Tfrecords(ERA5Dataset):
    def __init__(self, input_dir=None, dest_dir=None,  sequence_length=20, sequences_per_file=128, norm="minmax"):
        """
        This class is used for converting pkl files to tfrecords
        args:
            input_dir            : str, the path to the PreprocessData directory which is parent directory of "Pickle"
                                   and "tfrecords" files directiory.
            sequence_length      : int, default is 20, the sequen length per sample
            sequences_per_file   : int, how many sequences/samples per tfrecord to be saved
            norm                 : str, normalization methods from Norm_data class ("minmax" or "znorm";
                                   default: "minmax")
        """
        self.input_dir = input_dir
        self.output_dir = dest_dir
        # if the output_dir does not exist, then create it
        os.makedirs(self.output_dir, exist_ok=True)
        # get metadata,includes the var_in, image height, width etc.
        self.metadata_fl = os.path.join(os.path.dirname(self.input_dir.rstrip("/")), "metadata.json")
        self.get_metadata(MetaData(json_file=self.metadata_fl))
        # Get the data split informaiton
        self.sequence_length = sequence_length
        if norm == "minmax" or norm == "znorm":
            self.norm = norm
        else:
            raise ValueError("norm should be either 'minmax' or 'znorm'")
        self.sequences_per_file = sequences_per_file
        self.write_sequence_file()

    def get_years_months(self):
        """
        Get the months in the datasplit_config
        Return : 
                two elements: each contains 1-dim array with the months set from data_split_config json file
        """
        self.months = []
        self.years_months = []
        # search for pickle names with pattern 'X_{}.pkl'for months
        self.years = [name for name in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, name))]
        # search for folder names from pickle folder to get years
        patt = "X_*.pkl"         
        for year in self.years:
            months_pkl_list = glob.glob(os.path.join(self.input_dir, year, patt))
            months_list = [int(m[-6:-4]) for m in months_pkl_list]
            self.months.extend(months_list)
            self.years_months.append(months_list)
        return self.years, list(set(self.months)), self.years_months

    def get_stats_file(self):
        """
        Get the corresponding statistics file
        """
        method = ERA5Pkl2Tfrecords.get_stats_file.__name__

        stats_file = os.path.join(os.path.dirname(self.input_dir), "statistics.json")
        print("Opening json-file: {0}".format(stats_file))
        if os.path.isfile(stats_file):
            with open(stats_file) as js_file:
                self.stats = json.load(js_file)
        else:
            raise FileNotFoundError("%{0}: Could not find statistic file '{1}'".format(method, stats_file))

    def get_metadata(self, md_instance):
        """
        This function gets the meta data that has been generated in data_process_step1. Here, we aim to extract
        the height and width information from it
        vars_in   : list(str), must be consistent with the list from DataPreprocessing_step1
        height    : int, the height of the image
        width     : int, the width of the image
        """
        method = ERA5Pkl2Tfrecords.get_metadata.__name__
        
        if not isinstance(md_instance, MetaData):
            raise ValueError("%{0}: md_instance-argument must be a MetaData class instance".format(method))

        if not hasattr(self, "metadata_fl"):
            raise ValueError("%{0}: MetaData class instance passed, but attribute metadata_fl is still missing.".format(method))

        try:
            self.height, self.width = md_instance.ny, md_instance.nx
            self.vars_in = md_instance.variables
        except:
            raise IOError("%{0}: Could not retrieve all required information from metadata-file '{0}'"
                          .format(method, self.metadata_fl))

    @staticmethod
    def save_tf_record(output_fname, sequences, t_start_points):
        """
        Save the sequences, and the corresponding timestamp start point to tfrecords
        args:
            output_frames    : str, the file names of the output
            sequences        : list or array, the sequences want to be saved to tfrecords,
                               [sequences,seq_len,height,width,channels]
            t_start_points   : datetime type in the list, the first timestamp for each sequence
                               [seq_len,height,width, channel], the len of t_start_points is the same as sequences
        """
        method = ERA5Pkl2Tfrecords.save_tf_record.__name__

        sequences = np.array(sequences)
        # sanity checks
        assert sequences.shape[0] == len(t_start_points), "%{0}: Lengths of sequence differs from length of t_start_points.".format(method)
        assert isinstance(t_start_points[0], datetime.datetime), "%{0}: Elements of t_start_points must be datetime-objects.".format(method)

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

    def init_norm_class(self):
        """
        Get normalization data class 
        """
        method = ERA5Pkl2Tfrecords.init_norm_class.__name__

        print("%{0}: Make use of default minmax-normalization.".format(method))
        # init normalization-instance
        self.norm_cls = Norm_data(self.vars_in)
        self.nvars = len(self.vars_in)
        # get statistics file
        self.get_stats_file()
        # open statistics file and feed it to norm-instance
        self.norm_cls.check_and_set_norm(self.stats, self.norm)

    def normalize_vars_per_seq(self, sequences):
        """
        Normalize all the variables for the sequences
        args:
            sequences: list or array, is the sequences need to be saved to tfrecorcd.
                       The shape should be [sequences_per_file,seq_length,height,width,nvars]
        Return:
            the normalized sequences
        """
        method = ERA5Pkl2Tfrecords.normalize_vars_per_seq.__name__

        assert len(np.array(sequences).shape) == 5, "%{0}: Length of sequence array must be 5.".format(method)
        # normalization should adpot the selected variables, here we used duplicated channel temperature variables
        sequences = np.array(sequences)
        # normalization
        for i in range(self.nvars):
            sequences[..., i] = self.norm_cls.norm_var(sequences[..., i], self.vars_in[i], self.norm)
        return sequences

    def read_pkl_and_save_tfrecords(self, year, month):
        """
        Read pickle files based on month, to process and save to tfrecords,
        args:
            year    : int, the target year to save to tfrecord
            month   : int, the target month to save to tfrecord 
        """
        method = ERA5Pkl2Tfrecords.read_pkl_and_save_tfrecords.__name__

        # Define the input_file based on the year and month
        self.input_file_year = os.path.join(self.input_dir, str(year))
        input_file = os.path.join(self.input_file_year, 'X_{:02d}.pkl'.format(month))
        temp_input_file = os.path.join(self.input_file_year, 'T_{:02d}.pkl'.format(month))

        self.init_norm_class()
        sequences = []
        t_start_points = []
        sequence_iter = 0

        try:
            with open(input_file, "rb") as data_file:
                X_train = pickle.load(data_file)
        except:
            raise IOError("%{0}: Could not read data from pickle-file '{1}'".format(method, input_file))

        try:
            with open(temp_input_file, "rb") as temp_file:
                T_train = pickle.load(temp_file)
        except:
            raise IOError("%{0}: Could not read data from pickle-file '{1}'".format(method, temp_input_file))

        # check to make sure that X_train and T_train have the same length
        assert (len(X_train) == len(T_train))

        X_possible_starts = [i for i in range(len(X_train) - self.sequence_length)]
        for X_start in X_possible_starts:
            X_end = X_start + self.sequence_length
            seq = X_train[X_start:X_end, ...]
            # recording the start point of the timestamps (already datetime-objects)
            t_start = ERA5Pkl2Tfrecords.ensure_datetime(T_train[X_start][0])
            seq = list(np.array(seq).reshape((self.sequence_length, self.height, self.width, self.nvars)))
            if not sequences:
                last_start_sequence_iter = sequence_iter
            sequences.append(seq)
            t_start_points.append(t_start)
            sequence_iter += 1

            if len(sequences) == self.sequences_per_file:
                # normalize variables in the sequences
                sequences = ERA5Pkl2Tfrecords.normalize_vars_per_seq(self, sequences)
                output_fname = 'sequence_Y_{}_M_{}_{}_to_{}.tfrecords'.format(year, month, last_start_sequence_iter,
                                                                              sequence_iter - 1)
                output_fname = os.path.join(self.output_dir, output_fname)
                # write to tfrecord
                ERA5Pkl2Tfrecords.write_seq_to_tfrecord(output_fname, sequences, t_start_points)
                t_start_points = []
                sequences = []
        print("%{0}: Finished processing of input file '{1}'".format(method, input_file))

#         except FileNotFoundError as fnf_error:
#             print(fnf_error)

    @staticmethod
    def write_seq_to_tfrecord(output_fname, sequences, t_start_points):
        """
        Function to check if the sequences has been processed.
        If yes, the sequences are skipped, otherwise the sequences are saved to the output file
        """
        method = ERA5Pkl2Tfrecords.write_seq_to_tfrecord.__name__

        if os.path.isfile(output_fname):
            print("%{0}: TFrecord-file {1} already exists. It is therefore skipped.".format(method, output_fname))
        else:
            ERA5Pkl2Tfrecords.save_tf_record(output_fname, list(sequences), t_start_points)

    def write_sequence_file(self):
        """
        Generate a txt file, with the numbers of sequences for each tfrecords file.
        This is mainly used for calculting the number of samples for each epoch during training epoch
        """

        with open(os.path.join(self.output_dir, 'number_sequences.txt'), 'w') as seq_file:
            seq_file.write("%d\n" % self.sequences_per_file)


    @staticmethod
    def ensure_datetime(date):
        """
        Wrapper to return a datetime-object
        """
        method = ERA5Pkl2Tfrecords.ensure_datetime.__name__

        fmt = "%Y%m%d %H:%M"
        if isinstance(date, datetime.datetime):
            date_new = date
        else:
            try:
                date_new=pd.to_datetime(date)
                date_new=date_new.to_pydatetime()
            except Exception as err:
                print("%{0}: Could not handle input data {1} which is of type {2}.".format(method, date, type(date)))
                raise err

        return date_new

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))   
