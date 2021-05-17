"""
Class and functions toread data_split.json and hparams.json for gzprcp_data
"""
__email__ = "y.ji@fz-juelich.de"
__author__ = "Yan Ji, Bing Gong"
__date__ = "2021-05-09"


import glob
import os
import random
import json
import tensorflow as tf
from tensorflow.contrib.training import HParams
from collections import OrderedDict
from google.protobuf.json_format import MessageToDict

class GZprcp(object):
    def __init__(self, input_dir=None, datasplit_config=None, hparams_dict_config=None, mode="train",seed=None):
        """
        This class is used for preparing the data for gzprcp_data, and split the data to train/val/testing
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
        self.get_tfrecords_filename_base_datasplit()
        self.get_example_info()

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
        :return:
        """
        hparams = dict(
            context_frames=20,
            sequence_length=40,
            max_epochs = 20,
            batch_size = 4,
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
       self.all_filenames = glob.glob(os.path.join(self.input_dir,"*.tfrecords"))
       print("self.all_files",self.all_filenames)
       for indice_group, index in self.data_mode.items():
           fs = [GZprcp.string_filter(max_value=index[1], min_value=index[0], string=s) for s in self.all_filenames]
           self.tf_names = [self.all_filenames[fs_index] for fs_index in range(len(fs)) if fs[fs_index]==True]
           print("tf_names:",self.tf_names)
           print("tf_length",len(self.tf_names))
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
            raise ("The tfrecords pattern does not match the expected pattern, for instanct: 'sequence_index_0_to_10.tfrecords'") 
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
        self.video_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['sequence_length','height', 'width', 'channels'])
        self.image_shape = self.video_shape[1:]


    def num_examples_per_epoch(self):
        """
        Calculate how many tfrecords samples in the train/val/test
        """
        #count how many tfrecords files for train/val/testing
        len_fnames = len(self.filenames)
        seq_len_file = os.path.join(self.input_dir, 'number_sequences.txt')
        with open(seq_len_file, 'r') as sequence_lengths_file:
             sequence_lengths = sequence_lengths_file.readlines()
        sequence_lengths = [int(sequence_length.strip()) for sequence_length in sequence_lengths]
        self.num_examples_per_epoch  = len_fnames * sequence_lengths[0]
        return self.num_examples_per_epoch


    def make_dataset(self, batch_size):
        """
        Prepare batch_size dataset fed into to the models.
        If the data are from training dataset,then the data is shuffled;
        If the data are from val dataset, the shuffle var will be decided by the hparams.shuffled_on_val;
        if the data are from test dataset, the data will not be shuffled
        args:
              batch_size: int, the size of samples fed into the models per iteration
        """
        self.num_epochs = self.hparams.max_epochs
        def parser(serialized_example):
            seqs = OrderedDict()
            keys_to_features = {
                 'width': tf.FixedLenFeature([], tf.int64),
                 'height': tf.FixedLenFeature([], tf.int64),
                 'sequence_length': tf.FixedLenFeature([], tf.int64),
                 'channels': tf.FixedLenFeature([],tf.int64),
                 't_start': tf.FixedLenFeature([],tf.int64),
                 'images/encoded': tf.VarLenFeature(tf.float32)
             }
            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
            print("Image shape {}, {},{},{}".format(self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]))
            images = tf.reshape(seq, [self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]], name = "reshape_new")
            seqs["images"] = images
            return seqs
        filenames = self.filenames
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        print("number of epochs",self.num_epochs)
        if shuffle:
            random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames, buffer_size = 8* 1024 * 1024)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size =1024, count=self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)
        if self.mode == "val": dataset = dataset.repeat(110)
        num_parallel_calls = None if shuffle else 1
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        dataset = dataset.prefetch(batch_size)
        return dataset

    def make_batch(self, batch_size):
        dataset = self.make_dataset(batch_size)
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

