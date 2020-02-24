import argparse
import glob
import itertools
import os
import pickle
import random
import re
import netCDF4
import hickle as hkl
import numpy as np
import tensorflow as tf
import pandas as pd
from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
from collections import OrderedDict
from tensorflow.contrib.training import HParams

units = "hours since 2000-01-01 00:00:00"
calendar = "gregorian"

class ERA5Dataset_v2_anomaly(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(ERA5Dataset_v2_anomaly, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['height', 'width', 'channels'])
        self.state_like_names_and_shapes['images'] = 'images/encoded', image_shape

    def get_default_hparams_dict(self):
        default_hparams = super(ERA5Dataset_v2_anomaly, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,
            sequence_length=20,
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


    def filter(self, serialized_example):
        return tf.convert_to_tensor(True)



    def make_dataset_v2(self, batch_size):
        def parser(serialized_example):
            seqs = OrderedDict()
            keys_to_features = {
                # 'width': tf.FixedLenFeature([], tf.int64),
                # 'height': tf.FixedLenFeature([], tf.int64),
                'sequence_length': tf.FixedLenFeature([], tf.int64),
                # 'channels': tf.FixedLenFeature([],tf.int64),
                # 'images/encoded':  tf.FixedLenFeature([], tf.string)
                'images/encoded': tf.VarLenFeature(tf.float32)
            }
            # for i in range(20):
            #     keys_to_features["frames/{:04d}".format(i)] = tf.FixedLenFeature((), tf.string)
            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
            images = []
            # for i in range(20):
            #    images.append(parsed_features["images/encoded"].values[i])
            # images = parsed_features["images/encoded"]
            # images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["images/encoded"].values[i]),offsets)
            # seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"], '')
            # Parse the string into an array of pixels corresponding to the image
            # images = tf.decode_raw(parsed_features["images/encoded"],tf.int32)

            # images = seq
            images = tf.reshape(seq, [20, 64, 64, 1], name = "reshape_new")
            seqs["images"] = images
            return seqs
        filenames = self.filenames
        filenames_mean = self.filenames_mean
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        if shuffle:
            random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames, buffer_size = 8 * 1024 * 1024)  # todo: what is buffer_size
        dataset = dataset.filter(self.filter)
        #Bing: for Anomaly
        dataset_mean = tf.data.TFRecordDataset(filenames_mean, buffer_size = 8 * 1024 * 1024)
        dataset_mean = dataset_mean.filter(self.filter)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size = 1024, count = self.num_epochs))
            dataset_mean = dataset_mean.apply(tf.contrib.data.shuffle_and_repeat(buffer_size = 1024, count = self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)
            dataset_mean = dataset_mean.repeat(self.num_epochs)

        num_parallel_calls = None if shuffle else 1
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        dataset_mean = dataset_mean.apply(tf.contrib.data.map_and_batch(
            parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        #dataset = dataset.map(parser)
        # num_parallel_calls = None if shuffle else 1  # for reproducibility (e.g. sampled subclips from the test set)
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #    _parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls)) #  Bing: Parallel data mapping, num_parallel_calls normally depends on the hardware, however, normally should be equal to be the usalbe number of CPUs
        dataset = dataset.prefetch(batch_size)  # Bing: Take the data to buffer inorder to save the waiting time for GPU
        dataset_mean = dataset_mean.prefetch(batch_size)
        return dataset, dataset_mean

    def make_batch_v2(self, batch_size):
        dataset, dataset_mean = self.make_dataset_v2(batch_size)
        iterator = dataset.make_one_shot_iterator()
        interator2 = dataset_mean.make_one_shot_iterator()
        return iterator.get_next(), interator2.get_next()


    def make_data_mean(self,batch_size):
        pass



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


def extract_anomaly_one_pixel(X, X_timestamps,pixel):
    print("Processing Pixel {}, {}".format(pixel[0],pixel[1]))
    dates = [x.date() for x in X_timestamps]
    df = pd.DataFrame(data = X[:, pixel[0], pixel[1]], index = dates)
    df_mean = df.groupby(df.index).mean()
    df2 = pd.merge(df, df_mean, left_index = True, right_index = True)
    df2.columns = ["Real","Daily_mean"]
    df2["Anomaly"] = df2["Real"] - df2["Daily_mean"]
    daily_mean = df2["Daily_mean"].values
    anomaly = df2["Anomaly"].values
    return daily_mean, anomaly

def extract_anomaly_all_pixels(X, X_timestamps):
    #daily_mean, anomaly = extract_anomaly_one_pixel(X, X_timestamps, pixel = [0, 0])
    daily_mean_pixels = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    anomaly_pixels = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    #daily_mean_all_pixels = [extract_anomaly_one_pixel(X, X_timestamps, pixel = [i, j])[0] for i in range(X.shape[1]) for j in range(X.shape[2])]
    #anomaly_all_pixels = [extract_anomaly_one_pixel(X, X_timestamps, pixel = [i, j])[1] for i in range(X.shape[1]) for j in range(X.shape[2])]
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            daily_mean, anomaly = extract_anomaly_one_pixel(X, X_timestamps, pixel = [i, j])
            daily_mean_pixels[:,i,j] = daily_mean
            anomaly_pixels[:,i,j] = anomaly
    return daily_mean_pixels, anomaly_pixels


def read_frames_and_save_tf_records(output_dir, input_dir, partition_name, N_seq, sequences_per_file=128):#Bing: original 128
    output_orig_dir = os.path.join(output_dir,partition_name + "_orig")
    output_time_dir = os.path.join(output_dir,partition_name + "_time")
    output_mean_dir = os.path.join(output_dir,partition_name + "_mean")
    output_anomaly_dir = os.path.join(output_dir, partition_name )


    if not os.path.exists(output_orig_dir): os.mkdir(output_orig_dir)
    if not os.path.exists(output_time_dir): os.mkdir(output_time_dir)
    if not os.path.exists(output_mean_dir): os.mkdir(output_mean_dir)
    if not os.path.exists(output_anomaly_dir): os.mkdir(output_anomaly_dir)
    sequences = []
    sequences_time = []
    sequences_mean = []
    sequences_anomaly = []

    sequence_iter = 0
    sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
    X_train = hkl.load(os.path.join(input_dir, "X_" + partition_name + ".hkl"))
    X_time = hkl.load(os.path.join(input_dir, "Time_time_" + partition_name + ".hkl"))
    print ("X shape", X_train.shape)
    X_timestamps = [netCDF4.num2date(x, units = units, calendar = calendar) for x in X_time]

    print("X_time example", X_time[:10])
    print("X_time after to date", X_timestamps[:10])
    daily_mean_all_pixels, anomaly_all_pixels = extract_anomaly_all_pixels(X_train, X_timestamps)

    X_possible_starts = [i for i in range(len(X_train) - N_seq)]
    for X_start in X_possible_starts:
        print("Interation", sequence_iter)
        X_end = X_start + N_seq
        #seq = X_train[X_start:X_end, :, :,:]
        seq = X_train[X_start:X_end,:,:]
        seq_time = X_time[X_start:X_end]
        seq_mean = daily_mean_all_pixels[X_start:X_end,:,:]
        seq_anomaly = anomaly_all_pixels[X_start:X_end,:,:]
        #print("*****len of seq ***.{}".format(len(seq)))
        seq = list(np.array(seq).reshape((len(seq), 64, 64, 1)))
        seq_time = list(np.array(seq_time))
        seq_mean = list(np.array(seq_mean).reshape((len(seq_mean), 64, 64, 1)))
        seq_anomaly = list(np.array(seq_anomaly).reshape((len(seq_anomaly), 64, 64, 1)))
        if not sequences:
            last_start_sequence_iter = sequence_iter
            print("reading sequences starting at sequence %d" % sequence_iter)
        sequences.append(seq)
        sequences_time.append(seq_time)
        sequences_mean.append(seq_mean)
        sequences_anomaly.append(seq_anomaly)
        sequence_iter += 1
        sequence_lengths_file.write("%d\n" % len(seq))

        if len(sequences) == sequences_per_file:
            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter - 1)
            output_orig_fname = os.path.join(output_orig_dir, output_fname)
            output_time_fname = os.path.join(output_time_dir,'sequence_{0}_to_{1}.hkl'.format(last_start_sequence_iter, sequence_iter - 1))
            output_mean_fname = os.path.join(output_mean_dir, output_fname)
            output_anomaly_fname = os.path.join(output_anomaly_dir, output_fname)

            save_tf_record(output_orig_fname, sequences)
            hkl.dump(sequences_time,output_time_fname )
            #save_tf_record(output_time_fname,sequences_time)
            save_tf_record(output_mean_fname, sequences_mean)
            save_tf_record(output_anomaly_fname, sequences_anomaly)
            sequences[:] = []
            sequences_time[:] = []
            sequences_mean[:] = []
            sequences_anomaly[:] = []
    sequence_lengths_file.close()

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
        #ead_frames_and_save_tf_records(output_dir = output_dir, input_dir = input_dir,partition_name = partition_name, N_seq=20) #Bing: TODO: first try for N_seq is 10, but it met loading data issue. let's try 5

if __name__ == '__main__':
    main()

