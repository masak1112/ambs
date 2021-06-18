__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"

import argparse
import os
import glob
import random
import json
import tensorflow as tf
from collections import OrderedDict
from tensorflow.contrib.training import HParams
from google.protobuf.json_format import MessageToDict


class ERA5Dataset(object):

    def __init__(self,input_dir=None,datasplit_config=None,hparams_dict_config=None, mode='train',seed=None):
        """
        This class is used for preparing data for training/validation and test models
        args:
            input_dir            : the path of tfrecords files
            datasplit_config     : the path pointing to the datasplit_config json file
            hparams_dict_config  : the path to the dict that contains hparameters,
            mode                 : string, "train","val" or "test"
            seed                 : int, the seed for dataset 
        """
       # super(ERA5Dataset, self).__init__(**kwargs)
        self.input_dir = input_dir
        self.datasplit_config = datasplit_config
        self.mode = mode
        self.seed = seed
        self.sequence_length = None                             # will be set in get_example_info
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir)
        self.datasplit_dict_path = datasplit_config
        self.data_dict = self.get_datasplit()
        self.hparams_dict_config = hparams_dict_config
        self.hparams_dict = self.get_model_hparams_dict()
        self.hparams = self.parse_hparams() 
        self.get_tfrecords_filesnames_base_datasplit()
        self.get_example_info()

    def get_model_hparams_dict(self):
        """
        Get model_hparams_dict from json file
        """
        self.model_hparams_dict_load = {}
        if self.hparams_dict_config:
            with open(self.hparams_dict_config) as f:
                self.model_hparams_dict_load.update(json.loads(f.read()))
        return self.model_hparams_dict_load

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())


    def get_default_hparams_dict(self):
        """
        The function that contains default hparams
        Returns:
            A dict with the following hyperparameters.
            context_frames  : the number of ground-truth frames to pass in at start.
            max_epochs      : the number of epochs to train model
            lr              : learning rate
            loss_fun        : the loss function
        """
        hparams = dict(
            context_frames=10,
            max_epochs = 20,
            batch_size = 40,
            lr = 0.001,
            loss_fun = "rmse",
            shuffle_on_val= True,
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
        self.hparams_dict = reduce_dict(self.hparams_dict, self.get_default_hparams())
        parsed_hparams = self.get_default_hparams().override_from_dict(self.hparams_dict or {})
        return parsed_hparams

      
    def get_tfrecords_filesnames_base_datasplit(self):
        """
        Get  absolute .tfrecord path names based on the data splits patterns
        """
        self.filenames = []
        self.data_mode = self.data_dict[self.mode]
        self.tf_names = []
        for year, months in self.data_mode.items():
            for month in months:
                tf_files = "sequence_Y_{}_M_{}_*_to_*.tfrecord*".format(year,month)    
                self.tf_names.append(tf_files)
        # look for tfrecords in input_dir and input_dir/mode directories
        for files in self.tf_names:
            self.filenames.extend(glob.glob(os.path.join(self.input_dir, files)))
        if self.filenames:
            self.filenames = sorted(self.filenames)  # ensures order is the same across systems
        if not self.filenames:
            raise FileNotFoundError('No tfrecords were found in %s' % self.input_dir)


    def get_example_info(self):
        """
        Get the data information from an example tfrecord file
        """
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        print("features in dataset:",feature.keys())
        video_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['sequence_length', 'height',
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
        self.num_examples_per_epoch  = len_fnames * num_sequences[0]
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
                 't_start':  tf.VarLenFeature(tf.int64),
                 'images/encoded': tf.VarLenFeature(tf.float32)
             }

            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
            T_start = tf.sparse_tensor_to_dense(parsed_features["t_start"])
            print("Image shape {}, {},{},{}".format(self.sequence_length, self.image_shape[0], self.image_shape[1],
                                                    self.image_shape[2]))
            images = tf.reshape(seq, [self.sequence_length,self.image_shape[0],self.image_shape[1],
                                      self.image_shape[2]], name="reshape_new")
            seqs["images"] = images
            seqs["T_start"] = T_start
            return seqs
        filenames = self.filenames
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        if shuffle:
            random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames, buffer_size = 8* 1024 * 1024) 
        #dataset = dataset.filter(self.filter)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size =1024, count = self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        if self.mode == "val": dataset = dataset.repeat(20) 

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


