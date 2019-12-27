import argparse
import glob
import itertools
import os
import pickle
import random
import re
import hickle as hkl
import numpy as np
import tensorflow as tf
from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
#from base_dataset import VarLenFeatureVideoDataset
class ERA5Dataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(ERA5Dataset, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['height', 'width', 'channels'])
        self.state_like_names_and_shapes['images'] = 'images/encoded', image_shape

    def get_default_hparams_dict(self):
        default_hparams = super(ERA5Dataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,#Bing: Todo oriignal is 10
            sequence_length=20,#bing: TODO original is 20,
            long_sequence_length=40,
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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

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

def read_frames_and_save_tf_records(output_dir,input_dir,partition_name,N_seq,sequences_per_file=128):#Bing: original 128
    output_dir =  os.path.join(output_dir,partition_name)
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    sequences = []
    sequence_iter = 0
    X_train = hkl.load(os.path.join(input_dir, "X_" + partition_name + ".hkl"))
    print ("X shape", X_train.shape)
    X_possible_starts = [i for i in range(len(X_train) - N_seq)]
    for X_start in X_possible_starts:
        print("Interation", sequence_iter)
        X_end = X_start + N_seq
        #seq = X_train[X_start:X_end, :, :,:]
        seq = X_train[X_start:X_end,:,:]
        #print("*****len of seq ***.{}".format(len(seq)))
        seq = list(np.array(seq).reshape((len(seq), 64, 64, 1)))

        if not sequences:
            last_start_sequence_iter = sequence_iter
            print("reading sequences starting at sequence %d" % sequence_iter)
        sequences.append(seq)
        sequence_iter += 1

        if len(sequences) == sequences_per_file:
            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter - 1)
            output_fname = os.path.join(output_dir, output_fname)
            save_tf_record(output_fname, sequences)
            sequences[:] = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories ""boxing, handclapping, handwaving, ""jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    # parser.add_argument("image_size_h", type=int)
    # parser.add_argument("image_size_v", type = int)
    args = parser.parse_args()
    current_path = os.getcwd()
    #input_dir = "/Users/gongbing/PycharmProjects/video_prediction/splits"
    #output_dir = "/Users/gongbing/PycharmProjects/video_prediction/data/era5"
    partition_names = ['train', 'val', 'test']
    for partition_name in partition_names:
        read_frames_and_save_tf_records(output_dir=args.output_dir,input_dir=args.input_dir,partition_name=partition_name, N_seq=20) #Bing: Todo need check the N_seq
        #ead_frames_and_save_tf_records(output_dir = output_dir, input_dir = input_dir,partition_name = partition_name, N_seq=20) #Bing: TODO: first try for N_seq is 10, but it met loading data issue. let's try 5 tjen


if __name__ == '__main__':
    main()