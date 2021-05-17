"""
Class and functions required for preprocessing guizhou prcp data from .nc to TFRecords
"""
__email__ = "y.ji@fz-juelich.de"
__author__ = "Yan Ji, Bing Gong"
__date__ = "2021_05_09"


import os
import numpy as np
import tensorflow as tf
import argparse
import netCDF4 as nc
from model_modules.video_prediction.datasets.gzprcp_data import GZprcp


class GZprcp2Tfrecords(GZprcp):

    def __init__(self, input_dir=None, dest_dir=None, sequences_per_file=10):
        """
        This class is used for converting .nc files to tfrecords

        :param input_dir: str, the path direcotry to the file of npz
        :param dest_dir: the output  directory to save TFrecords.
        :param sequence_length: int, default is 40, the sequence length per sample
        :param sequences_per_file:int, how many sequences/samples per tfrecord to be saved
        """
        self.input_dir = input_dir
        self.output_dir = dest_dir
        os.makedirs(self.output_dir, exist_ok = True)
        self.sequences_per_file = sequences_per_file
        self.write_sequence_file()

    def __call__(self):
        """
        steps to process nc file to tfrecords
        :return: None
        """
        self.read_nc_file()
        self.save_nc_to_tfrecords()

    def read_nc_file(self):
        data_temp = nc.Dataset(os.path.join(self.input_dir, "guizhou_prcp.nc"))
        prcp_temp = np.transpose(data_temp['prcp'],[3,2,1,0])
        prcp_temp[np.isnan(prcp_temp)] = 0
        self.data = prcp_temp
        self.time = np.transpose(data_temp['time'],[2,1,0])
        print("data in gzprcp_test_Seq shape", self.data.shape)
        return None

    def save_nc_to_tfrecords(self):
        """
        Read the gzprcp data which is nc format, and save it to tfrecords files
        The shape of data_nc is [number_samples,seq_length,height,width]
        moving_mnst only has one channel
        """
        idx = 0
        num_samples = self.data.shape[0]
        if len(self.data.shape) == 4:
            #add one dim to represent channel, then got [num_samples,seq_length,height,width,channel]
            self.data = np.expand_dims(self.data, axis = 4)
        elif len(self.data.shape) == 5:
            pass
        else:
            #print('data shape nor match')
            raise (f"The shape of input movning mnist npz file is {len(self.data.shape)} which is not either 4 or 5, please further check your data source!")

        self.data = self.data.astype(np.float32)
        # self.data/= 255.0  # normalize RGB codes by dividing it to the max RGB value
        while idx < num_samples - self.sequences_per_file:
            sequences = self.data[idx:idx+self.sequences_per_file, :, :, :, :]
            t_start = self.time[idx,0,4]+self.time[idx,0,3]*100+self.time[idx,0,2]*10000+self.time[idx,0,1]*1000000+self.time[idx,0,0]*100000000
            output_fname = 'sequence_index_{}_to_{}.tfrecords'.format(idx, idx + self.sequences_per_file-1)
            output_fname = os.path.join(self.output_dir, output_fname)
            GZprcp2Tfrecords.save_tf_record(output_fname, sequences, t_start)
            idx = idx + self.sequences_per_file
        return None

    @staticmethod
    def save_tf_record(output_fname, sequences, t_start):
        with tf.python_io.TFRecordWriter(output_fname) as writer:
            for i in range(np.array(sequences).shape[0]):
                sequence = sequences[i, :, :, :, :]
                num_frames = len(sequence)
                height, width = sequence[0, :, :, 0].shape
                encoded_sequence = np.array([list(image) for image in sequence])
                features = tf.train.Features(feature = {
                    'sequence_length': _int64_feature(num_frames),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'channels': _int64_feature(1),
                    't_start': _int64_feature(int(t_start)),
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-source_dir", type=str, help="The input directory that contains the zgprcp_data nc file", default="/p/scratch/deepacf/video_prediction_shared_folder/extractedData/guizhou_prcp")
    parser.add_argument("-dest_dir", type=str,default="/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/gzprcp_data/tfrecords_seq_len_40")
    parser.add_argument("-sequences_per_file", type=int, default=10)
    args = parser.parse_args()
    inst = GZprcp2Tfrecords(args.source_dir, args.dest_dir, args.sequences_per_file)
    inst()


if __name__ == '__main__':
     main()


