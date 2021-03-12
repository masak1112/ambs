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
import json
import tensorflow as tf
from normalization import Norm_data
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
        # ML: No hidden path-extensions (rather managed in generate_runscript.py)
        # self.input_dir_pkl = os.path.join(input_dir,"pickle")
        self.output_dir = dest_dir
        # if the output_dir is not exist, then create it
        os.makedirs(self.output_dir, exist_ok=True)
        # get metadata,includes the var_in, image height, width etc.
        self.get_metadata()
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
            print("pahtL:", os.path.join(self.input_dir, year, patt))
            months_pkl_list = glob.glob(os.path.join(self.input_dir, year, patt))
            print("months_pkl_list", months_pkl_list)
            months_list = [int(m[-6:-4]) for m in months_pkl_list]
            self.months.extend(months_list)
            self.years_months.append(months_list)
        return self.years, list(set(self.months)), self.years_months

    def get_stats_file(self):
        """
        Get the corresponding statistics file
        """
        self.stats_file = os.path.join(os.path.dirname(self.input_dir), "statistics.json")
        print("Opening json-file: {0}".format(self.stats_file))
        if os.path.isfile(self.stats_file):
            with open(self.stats_file) as js_file:
                self.stats = json.load(js_file)
        else:
            raise FileNotFoundError("Statistic file does not exist")

    def get_metadata(self):
        """
        This function gets the meta data that has been generated in data_process_step1. Here, we aim to extract
        the height and width information from it
        vars_in   : list(str), must be consistent with the list from DataPreprocessing_step1
        height    : int, the height of the image
        width     : int, the width of the image
        """
        metadata_fl = os.path.join(os.path.dirname(self.input_dir.rstrip("/")), "metadata.json")
        if os.path.isfile(metadata_fl):
            self.metadata_fl = metadata_fl
            with open(self.metadata_fl) as f:
                self.metadata = json.load(f)
            self.frame_size = self.metadata["frame_size"]
            self.height = self.frame_size["ny"]
            self.width = self.frame_size["nx"]
            self.variables = self.metadata["variables"]
            self.vars_in = [list(var.values())[0] for var in self.variables]
           
        else:
            raise FileNotFoundError("The metadata file '{0}' was not generated properly, ".format(metadata_fl) +
                                    "you might need to re-run previous step of the workflow")

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
        sequences = np.array(sequences)
        # sanity checks
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

    def init_norm_class(self):
        """
        Get normalization data class 
        """
        print("Make use of default minmax-normalization...")
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
        assert len(np.array(sequences).shape) == 5
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
        # Define the input_file based on the year and month
        self.input_file_year = os.path.join(self.input_dir, str(year))
        input_file = os.path.join(self.input_file_year, 'X_{:02d}.pkl'.format(month))
        temp_input_file = os.path.join(self.input_file_year, 'T_{:02d}.pkl'.format(month))

        self.init_norm_class()
        sequences = []
        t_start_points = []
        sequence_iter = 0

        # try:
        with open(input_file, "rb") as data_file:
            X_train = pickle.load(data_file)
        with open(temp_input_file, "rb") as temp_file:
            T_train = pickle.load(temp_file)

        # check to make sure that X_train and T_train have the same length
        assert (len(X_train) == len(T_train))

        X_possible_starts = [i for i in range(len(X_train) - self.sequence_length)]
        for X_start in X_possible_starts:
            X_end = X_start + self.sequence_length
            seq = X_train[X_start:X_end, ...]
            # recording the start point of the timestamps
            t_start = T_train[X_start][0]
            t_start = datetime.datetime.strptime(str(t_start), "%Y-%m-%d %H:%M:%S")
            print("t_start,", t_start)
            print("type of t_starty", type(t_start))
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
            print("Finished for input file", input_file)

#         except FileNotFoundError as fnf_error:
#             print(fnf_error)

    @staticmethod
    def write_seq_to_tfrecord(output_fname, sequences, t_start_points):
        """
        Function to check if the sequences has been processed.
        If yes, the sequences are skipped, otherwise the sequences are saved to the output file
        """
        if os.path.isfile(output_fname):
            print(output_fname, 'already exists, skip it')
        else:
            ERA5Pkl2Tfrecords.save_tf_record(output_fname, list(sequences), t_start_points)

    def write_sequence_file(self):
        """
        Generate a txt file, with the numbers of sequences for each tfrecords file.
        This is mainly used for calculting the number of samples for each epoch during training epoch
        """

        with open(os.path.join(self.output_dir, 'number_sequences.txt'), 'w') as seq_file:
            seq_file.write("%d\n" % self.sequences_per_file)

#     def num_examples_per_epoch(self):
#         with open(os.path.join(self.input_dir, 'sequence_lengths.txt'), 'r') as sequence_lengths_file:
#             sequence_lengths = sequence_lengths_file.readlines()
#         sequence_lengths = [int(sequence_length.strip()) for sequence_length in sequence_lengths]
#         return np.sum(np.array(sequence_lengths) >= self.hparams.sequence_length)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))   
