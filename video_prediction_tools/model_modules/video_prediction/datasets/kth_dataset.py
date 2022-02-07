# SPDX-FileCopyrightText: 2018, alexlee-gk
#
# SPDX-License-Identifier: MIT

import argparse
import glob
import itertools
import os
import pickle
import random
import re
import tensorflow as tf
import numpy as np
import skimage.io
from collections import OrderedDict
from tensorflow.contrib.training import HParams
from google.protobuf.json_format import MessageToDict


class KTHVideoDataset(object):
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
        self.input_dir = input_dir
        self.datasplit_config = datasplit_config
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
        self.get_tfrecords_filesnames_base_datasplit()
        self.get_example_info()



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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def partition_data(input_dir):
    # List files and corresponding person IDs
    fnames = glob.glob(os.path.join(input_dir, '*/*'))
    fnames = [fname for fname in fnames if os.path.isdir(fname)]
    print("frames",fnames[0])
    persons = [re.match('person(\d+)_\w+_\w+', os.path.split(fname)[1]).group(1) for fname in fnames]
    persons = np.array([int(person) for person in persons])
    train_mask = persons <= 16
    train_fnames = [fnames[i] for i in np.where(train_mask)[0]]
    test_fnames = [fnames[i] for i in np.where(~train_mask)[0]]
    random.shuffle(train_fnames)
    pivot = int(0.95 * len(train_fnames))
    train_fnames, val_fnames = train_fnames[:pivot], train_fnames[pivot:]
    return train_fnames, val_fnames, test_fnames


def save_tf_record(output_fname, sequences):
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence in sequences:
            num_frames = len(sequence)
            height, width, channels = sequence[0].shape
            encoded_sequence = [image.tostring() for image in sequence]
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'images/encoded': _bytes_list_feature(encoded_sequence),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())



    def read_frames_and_save_tf_records(output_dir, video_dirs, image_size, sequences_per_file=128):
        partition_name = os.path.split(output_dir)[1] #Get the folder name train, val or test
        sequences = []
        sequence_iter = 0
        sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
        for video_iter, video_dir in enumerate(video_dirs): #Interate group (e.g. walking) each person
            meta_partition_name = partition_name if partition_name == 'test' else 'train'
            meta_fname = os.path.join(os.path.split(video_dir)[0], '%s_meta%dx%d.pkl' %
                                      (meta_partition_name, image_size, image_size))
            with open(meta_fname, "rb") as f:
                data = pickle.load(f) # The data has 62 items, each item is a dict, with three keys.  "vid","n", and "files", Each file has 4 channels, each channel has n sequence images with 64*64 png

            vid = os.path.split(video_dir)[1]
            (d,) = [d for d in data if d['vid'] == vid]
            for frame_fnames_iter, frame_fnames in enumerate(d['files']):
                frame_fnames = [os.path.join(video_dir, frame_fname) for frame_fname in frame_fnames]
                frames = skimage.io.imread_collection(frame_fnames)
                # they are grayscale images, so just keep one of the channels
                frames = [frame[..., 0:1] for frame in frames]

                if not sequences: #The length of the sequence in sequences could be different
                    last_start_sequence_iter = sequence_iter
                    print("reading sequences starting at sequence %d" % sequence_iter)

                sequences.append(frames)
                sequence_iter += 1
                sequence_lengths_file.write("%d\n" % len(frames))

                if (len(sequences) == sequences_per_file or
                        (video_iter == (len(video_dirs) - 1) and frame_fnames_iter == (len(d['files']) - 1))):
                    output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter - 1)
                    output_fname = os.path.join(output_dir, output_fname)
                    save_tf_record(output_fname, sequences)
                    sequences[:] = []
        sequence_lengths_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories "
                                                    "boxing, handclapping, handwaving, "
                                                    "jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("image_size", type=int)
    args = parser.parse_args()
    partition_names = ['train', 'val', 'test']
    print("input dir", args.input_dir)
    partition_fnames = partition_data(args.input_dir)
    print("partiotion_fnames[0]", partition_fnames[0])
    for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)
        read_frames_and_save_tf_records(partition_dir, partition_fnames, args.image_size)


if __name__ == '__main__':
    main()
