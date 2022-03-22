# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Karim"
__date__ = "2021-05-03"

import glob
import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
from collections import OrderedDict
from google.protobuf.json_format import MessageToDict


class MovingMnist(object):
    def __init__(self, input_dir: str = None, datasplit_config: str = None, hparams_dict_config: str = None,
                 mode: str = "train", seed: int = None, nsamples_ref: int = None):
        """
        This class is used for preparing data for training/validation and test models
        :param input_dir: the path of tfrecords files
        :param datasplit_config: the path pointing to the datasplit_config json file
        :param hparams_dict_config: the path to the dict that contains hparameters,
        :param mode: string, "train","val" or "test"
        :param seed: int, the seed for dataset
        :param nsamples_ref: number of reference samples whch can be used to control repetition factor for dataset
                             for ensuring adopted size of dataset iterator (used for validation data during training)
                             Example: Let nsamples_ref be 1000 while the current datset consists 100 samples, then
                                      the repetition-factor will be 10 (i.e. nsamples*rep_fac = nsamples_ref)
        """
        method = self.__class__.__name__

        self.input_dir = input_dir
        self.mode = mode 
        self.seed = seed
        self.sequence_length = None                             # will be set in get_example_info
        self.shuffled = False                                   # will be set properly in make_dataset-method
        # sanity checks
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('%{0}: Invalid mode {1}'.format(method, self.mode))
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("%{0} input_dir '{1}' does not exist".format(method, self.input_dir))
        if nsamples_ref is not None:
            self.nsamples_ref = nsamples_ref
        self.datasplit_dict_path = datasplit_config
        self.data_dict = self.get_datasplit()
        self.hparams_dict_config = hparams_dict_config
        self.hparams_dict = self.get_model_hparams_dict()
        self.hparams = self.parse_hparams()
        self.get_tfrecords_filename_base_datasplit()
        self.get_example_info()

    def get_datasplit(self):
        """
        Get the datasplit json file
        """
        with open(self.datasplit_dict_path) as f:
            datasplit_dict = json.load(f)
        return datasplit_dict

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
        :return:
        """
        hparams = dict(
            context_frames=10,
            sequence_length=20,
            max_epochs=20,
            batch_size=40,
            lr=0.001,
            loss_fun="rmse",
            shuffle_on_val=True,
        )
        return hparams

    def get_tfrecords_filename_base_datasplit(self):
       """
       Get obsoluate .tfrecords names based on the data splits patterns
       """
       self.filenames = []
       self.data_mode = self.data_dict[self.mode]
       self.all_filenames = glob.glob(os.path.join(self.input_dir,"*.tfrecords"))
       print("self.all_files",self.all_filenames)
       for indice_group, index in self.data_mode.items():
           fs = [MovingMnist.string_filter(max_value=index[1], min_value=index[0], string=s) for s in self.all_filenames]
           print("fs:",fs)
           self.tf_names = [self.all_filenames[fs_index] for fs_index in range(len(fs)) if fs[fs_index]==True]
           print("tf_names,",self.tf_names)
       # look for tfrecords in input_dir and input_dir/mode directories
       for files in self.tf_names:
            self.filenames.extend(glob.glob(os.path.join(self.input_dir, files)))
       if self.filenames:
           self.filenames = sorted(self.filenames)  # ensures order is the same across systems
       if not self.filenames:
           raise FileNotFoundError('No tfrecords were found in %s' % self.input_dir)

    @staticmethod
    def string_filter(max_value=None, min_value=None, string="input_directory/sequence_index_0_index_10.tfrecords"):
        a = os.path.split(string)[-1].split("_")
        if not len(a) == 5:
            raise ("The tfrecords pattern does not match the expected pattern, for instance: 'sequence_index_0_to_10.tfrecords'")
        min_index = int(a[2])
        max_index = int(a[4].split(".")[0])
        if min_index >= min_value and max_index <= max_value:
            return True
        else:
            return False

    def get_example_info(self):
        """
         Get the data information from tfrecord file
        """
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        print("features in dataset:",feature.keys())
        video_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['sequence_length','height',
                                                                                        'width', 'channels'])
        self.sequence_length = video_shape[0]
        self.image_shape = video_shape[1:]

    def num_examples_per_epoch(self):
        """
        Calculate how many tfrecords samples in the train/val/test
        """
        # count how many tfrecords files for train/val/testing
        len_fnames = len(self.filenames)
        num_seq_file = os.path.join(self.input_dir, 'number_sequences.txt')
        with open(num_seq_file, 'r') as dfile:
             num_seqs = dfile.readlines()
        num_sequences = [int(num_seq.strip()) for num_seq in num_seqs]
        num_examples_per_epoch = len_fnames * num_sequences[0]

        return num_examples_per_epoch

    def make_dataset(self, batch_size):
        """
        Prepare batch_size dataset fed into to the models.
        If the data are from training dataset,then the data is shuffled;
        If the data are from val dataset, the shuffle var will be decided by the hparams.shuffled_on_val;
        if the data are from test dataset, the data will not be shuffled
        args:
              batch_size: int, the size of samples fed into the models per iteration
        """
        method = MovingMnist.make_dataset.__name__

        self.num_epochs = self.hparams.max_epochs

        def parser(serialized_example):
            seqs = OrderedDict()
            keys_to_features = {
                 'width': tf.FixedLenFeature([], tf.int64),
                 'height': tf.FixedLenFeature([], tf.int64),
                 'sequence_length': tf.FixedLenFeature([], tf.int64),
                 'channels': tf.FixedLenFeature([],tf.int64),
                 'images/encoded': tf.VarLenFeature(tf.float32)
             }
            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
            print("Image shape {}, {},{},{}".format(self.sequence_length,self.image_shape[0],self.image_shape[1],
                                                    self.image_shape[2]))
            images = tf.reshape(seq, [self.sequence_length,self.image_shape[0],self.image_shape[1],
                                      self.image_shape[2]], name = "reshape_new")
            seqs["images"] = images
            return seqs
        filenames = self.filenames
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        if shuffle:
            self.shuffled = True
            random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8*1024*1024)
        # set-up dataset iterator
        nrepeat = self.num_epochs
        if self.nsamples_ref:
            num_samples = self.num_examples_per_epoch()
            nrepeat = int(nrepeat*max(int(np.ceil(self.nsamples_ref/num_samples)), 1))

        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=nrepeat))
        else:
            dataset = dataset.repeat(nrepeat)
        num_parallel_calls = None if shuffle else 1
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        dataset = dataset.prefetch(batch_size)
        return dataset

    def make_batch(self, batch_size):
        dataset = self.make_dataset(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


# further auxiliary methods
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




    
