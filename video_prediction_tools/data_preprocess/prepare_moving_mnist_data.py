"""
Class and functions required for preprocessing Moving mnist data from .npz to TFRecords
"""
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Karim Mache"
__date__ = "2021_05_04"


import os
import numpy as np
import tensorflow as tf
from model_modules.video_prediction.datasets import moving_mnist


class MovingMnist2Tfrecords(moving_mnist):

    def __init__(self, input_dir=None, dest_dir=None,  sequence_length=20, sequences_per_file=128):
        """
        This class is used for converting .npz files to tfrecords

        :param input_dir: str, the path direcotry to the file of npz
        :param dest_dir: the output  directory to save TFrecords.
        :param sequence_length: int, default is 20, the sequence length per sample
        :param sequences_per_file:int, how many sequences/samples per tfrecord to be saved
        """
        self.input_dir = input_dir
        self.output_dir = dest_dir

        os.makedirs(self.output_dir, exist_ok = True)
        self.sequence_length = sequence_length
        self.sequences_per_file = sequences_per_file
        self.write_sequence_file()

    def read_npz_file(self):
        self.data = np.load(os.path.join(self.input_dir, "mnist_test_seq.npy"))
        print("data in minist_test_Seq shape", self.data.shape)
        return None

    def save_npz_to_tf_records(self, seq_length=20, sequences_per_file=20):  # Bing: original 128
        """
        Read the moving_mnst data which is npz format, and save it to tfrecords files
        The shape of dat_npz is [seq_length,number_samples,height,width]
        moving_mnst only has one channel

        """
        idx = 0
        num_samples = self.data.shape[1]
        if len(self.data.shape) == 4:
            #add one dim to represent channel, then got [seq_length,num_samples,height,width,channel]
            self.data = np.expand_dims(self.data, axis = 4)
        elif len(self.data.shape) == 5:
            pass
        else:
            raise (f"The shape of input movning mnist npz file is {len(self.data.shape)} which is not either 4 or 5, please further check your data source!")

        self.data = self.data.astype(np.float32)
        self.data/= 255.0  # normalize RGB codes by dividing it to the max RGB value
        while idx < num_samples - sequences_per_file:
            sequences = self.data[:, idx:idx + sequences_per_file, :, :, :]
            output_fname = 'sequence_index_{}_index_{}.tfrecords'.format(idx, idx + sequences_per_file)
            output_fname = os.path.join(self.output_dir, output_fname)
            MovingMnist2Tfrecords.save_tf_record(output_fname, sequences)
            idx = idx + sequences_per_file
        return None

    @staticmethod
    def save_tf_record(output_fname, sequences):
        with tf.python_io.TFRecordWriter(output_fname) as writer:
            for i in range(len(sequences)):
                sequence = sequences[:, i, :, :, :]
                num_frames = len(sequence)
                height, width = sequence[0, :, :, 0].shape
                encoded_sequence = np.array([list(image) for image in sequence])
                features = tf.train.Features(feature = {
                    'sequence_length': _int64_feature(num_frames),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'channels': _int64_feature(1),
                    'images/encoded': _floats_feature(encoded_sequence.flatten()),
                })
                example = tf.train.Example(features = features)
                writer.write(example.SerializeToString())

    def write_sequence_file(self):
        """
        Generate a txt file, with the numbers of sequences for each tfrecords file.
        This is mainly used for calculting the number of samples for each epoch during training epoch
        """

        with open(os.path.join(self.output_dir, 'number_sequences.txt'), 'w') as seq_file:
            seq_file.write("%d\n" % self.sequences_per_file)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))