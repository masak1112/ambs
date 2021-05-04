
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Karim"
__date__ = "2021-05-03"


import argparse
import glob
import os
import random
import numpy as np
import json
import tensorflow as tf
from tensorflow.contrib.training import HParams
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from google.protobuf.json_format import MessageToDict


class MovingMnist(object):
    def __init__(self, input_dir=None, datasplit_config=None, hparams_dict_config=None, mode="train",seed=None):
        """
        This class is used for preparing the data for moving mnist, and split the data to train/val/testing
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
        self.get_tfrecords_filesnames_base_datasplit()
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
            context_frames=10,
            sequence_length=20,
            max_epochs = 20,
            batch_size = 40,
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
       self.tf_names = []
       self.all_filenames = glob.glob(os.path.join(self.input_dir,"*.tfrecords"))
       for indice_group, index in self.data_mode.items():
           for indice in index:
               max_val = indice[1]
               min_val = indice[0]
               fs = [MovingMnist.string_filter(max_val=max_val, min_val=min_val, string=s) for s in self.all_filenames]
               self.tf_names.append(self.all_filenames[fs])

       # look for tfrecords in input_dir and input_dir/mode directories
       for files in self.tf_names:
            self.filenames.extend(glob.glob(os.path.join(self.input_dir, files)))
       if self.filenames:
           self.filenames = sorted(self.filenames)  # ensures order is the same across systems
       if not self.filenames:
           raise FileNotFoundError('No tfrecords were found in %s' % self.input_dir)


    @staticmethod
    def string_filter(max_value=None, min_value=None, string="input_directory/sequence_index_0_index_10.tfrecords"):

        a = os.path.split(string)[1].split("_")
        min_index = int(a[2])
        max_index = int(a[4])
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
        if shuffle:
            random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames, buffer_size = 8* 1024 * 1024)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size =1024, count=self.num_epochs))
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

def save_tf_record(output_fname, sequences):
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for i in range(len(sequences)):
            sequence = sequences[:,i,:,:,:] 
            num_frames = len(sequence)
            height, width = sequence[0,:,:,0].shape
            encoded_sequence = np.array([list(image) for image in sequence])
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(1),
                'images/encoded': _floats_feature(encoded_sequence.flatten()),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())




def write_sequence_file(output_dir,seq_length,sequences_per_file):    
    partition_names = ["train","val","test"]
    for partition_name in partition_names:
        save_output_dir = os.path.join(output_dir,partition_name)
        tfCounter = len(glob.glob1(save_output_dir,"*.tfrecords"))
        print("Partition_name: {}, number of tfrecords: {}".format(partition_name,tfCounter))
        sequence_lengths_file = open(os.path.join(save_output_dir, 'sequence_lengths.txt'), 'w')
        for i in range(tfCounter*sequences_per_file):
            sequence_lengths_file.write("%d\n" % seq_length)
        sequence_lengths_file.close()


def plot_seq_imgs(imgs,output_png_dir,idx,label="Ground Truth"):
    """
    Plot the seq images 
    """

    if len(np.array(imgs).shape)!=3:raise("img dims should be three: (seq_len,lat,lon)")
    img_len = imgs.shape[0]
    fig = plt.figure(figsize=(18,6))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace = 0., hspace = 0.)
    for i in range(img_len):
        ax1 = plt.subplot(gs[i])
        plt.imshow(imgs[i] ,cmap = 'jet')
        plt.setp([ax1], xticks = [], xticklabels = [], yticks = [], yticklabels = [])
    plt.savefig(os.path.join(output_png_dir, label + "_" +   str(idx) +  ".jpg"))
    print("images_saved")
    plt.clf()


    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories ""boxing, handclapping, handwaving, ""jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-sequences_per_file",type=int,default=2)
    args = parser.parse_args()
    current_path = os.getcwd()
    data = np.load(os.path.join(args.input_dir,"mnist_test_seq.npy"))
    print("data in minist_test_Seq shape",data.shape)
    seq_length =  data.shape[0]
    height = data.shape[2]
    width = data.shape[3]
    num_samples = data.shape[1] 
    max_npz = np.max(data)
    min_npz = np.min(data)
    print("max_npz,",max_npz)
    print("min_npz",min_npz)
    #Todo need to discuss how to split the data, since we have totally 10000 samples, the origin paper convLSTM used 10000 as training, 2000 as validation and 3000 for testing
    dat_train = data[:,:6000,:,:]
    dat_val = data[:,6000:7000,:,:]
    dat_test = data[:,7000:,:]
    #plot_seq_imgs(dat_test[10:,0,:,:],output_png_dir="/p/project/deepacf/deeprain/video_prediction_shared_folder/results/moving_mnist/convLSTM",idx=1,label="Ground Truth from npz")
    #save train
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"train"),dat_train, seq_length=20, sequences_per_file=40, height=height, width=width)
    #save val
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"val"),dat_val, seq_length=20, sequences_per_file=40, height=height, width=width)
    #save test     
    #read_frames_and_save_tf_records(os.path.join(args.output_dir,"test"),dat_test, seq_length=20, sequences_per_file=40, height=height, width=width)
    #write_sequence_file(output_dir=args.output_dir,seq_length=20,sequences_per_file=40)
if __name__ == '__main__':
     main()

