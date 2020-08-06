import argparse
import glob
import itertools
import os
import pickle
import random
import re
import hickle as hkl
import numpy as np
import json
import tensorflow as tf
from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
# ML 2020/04/14: hack for getting functions of process_netCDF_v2:
from os import path
import sys
sys.path.append(path.abspath('../../workflow_parallel_frame_prediction/'))
import DataPreprocess.process_netCDF_v2 
from DataPreprocess.process_netCDF_v2 import get_unique_vars
from DataPreprocess.process_netCDF_v2 import Calc_data_stat
#from base_dataset import VarLenFeatureVideoDataset
from collections import OrderedDict
from tensorflow.contrib.training import HParams
from mpi4py import MPI
import glob



class ERA5Dataset_v2(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(ERA5Dataset_v2, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        self.video_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['sequence_length','height', 'width', 'channels'])
        self.image_shape = self.video_shape[1:]
        self.state_like_names_and_shapes['images'] = 'images/encoded', self.image_shape

    def get_default_hparams_dict(self):
        default_hparams = super(ERA5Dataset_v2, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,#Bing: Todo oriignal is 10
            sequence_length=20,#bing: TODO original is 20,
            long_sequence_length=20,
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
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'sequence_length': tf.FixedLenFeature([], tf.int64),
                'channels': tf.FixedLenFeature([],tf.int64),
                # 'images/encoded':  tf.FixedLenFeature([], tf.string)
                'images/encoded': tf.VarLenFeature(tf.float32)
            }
            
            # for i in range(20):
            #     keys_to_features["frames/{:04d}".format(i)] = tf.FixedLenFeature((), tf.string)
            parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
            print ("Parse features", parsed_features)
            seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"])
           # width = tf.sparse_tensor_to_dense(parsed_features["width"])
           # height = tf.sparse_tensor_to_dense(parsed_features["height"])
           # channels  = tf.sparse_tensor_to_dense(parsed_features["channels"])
           # sequence_length = tf.sparse_tensor_to_dense(parsed_features["sequence_length"])
            images = []
            # for i in range(20):
            #    images.append(parsed_features["images/encoded"].values[i])
            # images = parsed_features["images/encoded"]
            # images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["images/encoded"].values[i]),offsets)
            # seq = tf.sparse_tensor_to_dense(parsed_features["images/encoded"], '')
            # Parse the string into an array of pixels corresponding to the image
            # images = tf.decode_raw(parsed_features["images/encoded"],tf.int32)

            # images = seq
            print("Image shape {}, {},{},{}".format(self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]))
            images = tf.reshape(seq, [self.video_shape[0],self.image_shape[0],self.image_shape[1], self.image_shape[2]], name = "reshape_new")
            seqs["images"] = images
            return seqs
        filenames = self.filenames



        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        if shuffle:
            random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames, buffer_size = 8* 1024 * 1024)  # todo: what is buffer_size
        print("files", self.filenames)
        print("mode", self.mode)
        dataset = dataset.filter(self.filter)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size =1024, count = self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        num_parallel_calls = None if shuffle else 1
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        #dataset = dataset.map(parser)
        # num_parallel_calls = None if shuffle else 1  # for reproducibility (e.g. sampled subclips from the test set)
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #    _parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls)) #  Bing: Parallel data mapping, num_parallel_calls normally depends on the hardware, however, normally should be equal to be the usalbe number of CPUs
        dataset = dataset.prefetch(batch_size)  # Bing: Take the data to buffer inorder to save the waiting time for GPU

        return dataset



    def make_batch(self, batch_size):
        dataset = self.make_dataset_v2(batch_size)
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
    #print('saving sequences to %s' % output_fname)
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
            
class Norm_data:
    """
     Class for normalizing data. The statistical data for normalization (minimum, maximum, average, standard deviation etc.) is expected to be available from a statistics-dictionary
     created with the calc_data_stat-class (see 'process_netCDF_v2.py'.
    """
    
    ### set known norms and the requested statistics (to be retrieved from statistics.json) here ###
    known_norms = {}
    known_norms["minmax"] = ["min","max"]
    known_norms["znorm"]  = ["avg","sigma"]
   
    def __init__(self,varnames):
        """Initialize the instance by setting the variable names to be handled and the status (for sanity checks only) as attributes."""
        varnames_uni, _, nvars = get_unique_vars(varnames)
        
        self.varnames = varnames_uni
        self.status_ok= False
            
    def check_and_set_norm(self,stat_dict,norm):
        """
         Checks if the statistics-dictionary provides the required data for selected normalization method and expands the instance's attributes accordingly.
         Example: minmax-normalization requires the minimum and maximum value of a variable named var1. 
                 If the requested values are provided by the statistics-dictionary, the instance gets the attributes 'var1min' and 'var1max',respectively.
        """
        
        # some sanity checks
        if not norm in self.known_norms.keys(): # valid normalization requested?
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '"+norm+"' is unknown.")
       
        if not all(items in stat_dict for items in self.varnames): # all variables found in dictionary?
            print("Keys in stat_dict:")
            print(stat_dict.keys())
            
            print("Requested variables:")
            print(self.varnames)
            raise ValueError("Could not find all requested variables in statistics dictionary.")   

        # create all attributes for the instance
        for varname in self.varnames:
            for stat_name in self.known_norms[norm]:
                #setattr(self,varname+stat_name,stat_dict[varname][0][stat_name])
                setattr(self,varname+stat_name,Calc_data_stat.get_stat_vars(stat_dict,stat_name,varname))
                
        self.status_ok = True           # set status for normalization -> ready
                
    def norm_var(self,data,varname,norm):
        """ 
         Performs given normalization on input data (given that the instance is already set up)
        """
        
        # some sanity checks
        if not self.status_ok: raise ValueError("Norm_data-instance needs to be initialized and checked first.") # status ready?
        
        if not norm in self.known_norms.keys():                                # valid normalization requested?
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '"+norm+"' is unknown.")
        
        # do the normalization and return
        if norm == "minmax":
            return((data[...] - getattr(self,varname+"min"))/(getattr(self,varname+"max") - getattr(self,varname+"min")))
        elif norm == "znorm":
            return((data[...] - getattr(self,varname+"avg"))/getattr(self,varname+"sigma")**2)
        
    def denorm_var(self,data,varname,norm):
        """ 
         Performs given denormalization on input data (given that the instance is already set up), i.e. inverse method to norm_var
        """
        
        # some sanity checks
        if not self.status_ok: raise ValueError("Norm_data-instance needs to be initialized and checked first.") # status ready?        
        
        if not norm in self.known_norms.keys():                                # valid normalization requested?
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '"+norm+"' is unknown.")
        
        # do the denormalization and return
        if norm == "minmax":
            return(data[...] * (getattr(self,varname+"max") - getattr(self,varname+"min")) + getattr(self,varname+"min"))
        elif norm == "znorm":
            return(data[...] * getattr(self,varname+"sigma")**2 + getattr(self,varname+"avg"))
        

def read_frames_and_save_tf_records(stats,output_dir,input_file,vars_in,year,month,seq_length=20,sequences_per_file=128,height=64,width=64,channels=3,**kwargs):#Bing: original 128
    """
    Read hickle/pickle files based on month, to process and save to tfrecords
    stats:dict, contains the stats information from hickle directory,
    input_file: string, absolute path to hickle/pickle file
    file_info: 1D list with three elements, partition_name(train,val or test), year, and month e.g.[train,1,2]  
    """
    # ML 2020/04/08:
    # Include vars_in for more flexible data handling (normalization and reshaping)
    # and optional keyword argument for kind of normalization
    print ("read_frames_and_save_tf_records function") 
    if 'norm' in kwargs:
        norm = kwargs.get("norm")
    else:
        norm = "minmax"
        print("Make use of default minmax-normalization...")
    
   
    os.makedirs(output_dir,exist_ok=True)
    
    norm_cls  = Norm_data(vars_in)       # init normalization-instance
    nvars     = len(vars_in)
    
    # open statistics file and feed it to norm-instance
    #with open(os.path.join(input_dir,"statistics.json")) as js_file:
    norm_cls.check_and_set_norm(stats,norm)        
    
    sequences = []
    sequence_iter = 0
    #sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
    # ML 2020/07/15: Make use of pickle-files only
    #with open(os.path.join(input_dir, "X_" + partition_name + ".pkl"), "rb") as data_file:
    #    X_train = pickle.load(data_file)
    #Bing 2020/07/16
    #print ("open intput dir,",input_file)
    with open(input_file, "rb") as data_file:
        X_train = pickle.load(data_file)
    
    #X_train = hkl.load(os.path.join(input_dir, "X_" + partition_name + ".hkl"))
    X_possible_starts = [i for i in range(len(X_train) - seq_length)]
    for X_start in X_possible_starts:
        
        X_end = X_start + seq_length
        #seq = X_train[X_start:X_end, :, :,:]
        seq = X_train[X_start:X_end,:,:,:]
        #print("*****len of seq ***.{}".format(len(seq)))
       
        seq = list(np.array(seq).reshape((seq_length, height, width, nvars)))
        if not sequences:
            last_start_sequence_iter = sequence_iter
            #print("reading sequences starting at sequence %d" % sequence_iter)
        sequences.append(seq)
        sequence_iter += 1
        #sequence_lengths_file.write("%d\n" % len(seq))

        if len(sequences) == sequences_per_file:
            ###Normalization should adpot the selected variables, here we used duplicated channel temperature variables
            sequences = np.array(sequences)
            ### normalization
            for i in range(nvars):    
                sequences[:,:,:,:,i] = norm_cls.norm_var(sequences[:,:,:,:,i],vars_in[i],norm)

            output_fname = 'sequence_Y_{}_M_{}_{}_to_{}.tfrecords'.format(year,month,last_start_sequence_iter,sequence_iter - 1)
            output_fname = os.path.join(output_dir, output_fname)
            save_tf_record(output_fname, list(sequences))
            sequences = []
    print("Finished for input file",input_file)
    #sequence_lengths_file.close()
    return 

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
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories ""boxing, handclapping, handwaving, ""jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    # ML 2020/04/08 S
    # Add vars for ensuring proper normalization and reshaping of sequences
    parser.add_argument("-vars","--variables",dest="variables", nargs='+', type=str, help="Names of input variables.")
    parser.add_argument("-height",type=int,default=64)
    parser.add_argument("-width",type = int,default=64)
    parser.add_argument("-seq_length",type=int,default=20)
    parser.add_argument("-sequences_per_file",type=int,default=2)
    args = parser.parse_args()
    current_path = os.getcwd()
    #input_dir = "/Users/gongbing/PycharmProjects/video_prediction/splits"
    #output_dir = "/Users/gongbing/PycharmProjects/video_prediction/data/era5"


      
    partition = {
            "train":{
               # "2222":[1,2,3,5,6,7,8,9,10,11,12],
               # "2010_1":[1,2,3,4,5,6,7,8,9,10,11,12],
               # "2012":[1,2,3,4,5,6,7,8,9,10,11,12],
               # "2013_complete":[1,2,3,4,5,6,7,8,9,10,11,12],
                "2015":[1,2,3,4,5,6,7,8,9,10,11,12],
               # "2017":[1,2,3,4,5,6,7,8,9,10,11,12]
                 },
            "val":
                {"2016":[1,2,3,4,5,6,7,8,9,10,11,12]
                 },
            "test":
                {"2017":[1,2,3,4,5,6,7,8,9,10,11,12]
                 }
            }
    
        # open statistics file and feed it to norm-instance
    with open(os.path.join(args.input_dir,"statistics.json")) as js_file:
        stats = json.load(js_file)
    
    #TODO: search all the statistic json file correspdoing to the parition and generate a general statistic.json for normalization


    # ini. MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()  # rank of the node
    p = comm.Get_size()  # number of assigned nods
    
    if my_rank == 0 :
        partition_year_month = [] #contain lists of list, each list includes three element [train,year,month]
        partition_names = list(partition.keys())
        print ("partition_names:",partition_names)
        broadcast_lists = []
        for partition_name in partition_names:
            partition_data = partition[partition_name]        
            years = list(partition_data.keys())
            broadcast_lists.append([partition_name,years])
        for nodes in range(1,p):
            #ibroadcast_list = [partition_name,years,nodes]
            #broadcast_lists.append(broadcast_list)
            comm.send(broadcast_lists,dest=nodes) 
           
        message_counter = 1
        while message_counter <= 12:
            message_in = comm.recv()
            message_counter = message_counter + 1 
            print("Message in from slaver",message_in) 
            
        write_sequence_file(args.output_dir,args.seq_length,args.sequences_per_file)
        
        #write_sequence_file   
    else:
        message_in = comm.recv()
        print ("My rank,", my_rank)   
        print("message_in",message_in)
        #loop the partitions (train,val,test)
        for partition in message_in:
            print("partition on slave ",partition)
            partition_name = partition[0]
            save_output_dir =  os.path.join(args.output_dir,partition_name)
            for year in partition[1]:
               input_file = "X_" + '{0:02}'.format(my_rank) + ".pkl"
               input_dir = os.path.join(args.input_dir,year)
               input_file = os.path.join(input_dir,input_file)
               read_frames_and_save_tf_records(year=year,month=my_rank,stats=stats,output_dir=save_output_dir,input_file=input_file,vars_in=args.variables,partition_name=partition_name, seq_length=args.seq_length,height=args.height,width=args.width,sequences_per_file=args.sequences_per_file)        
            print("Year {} finished",year)
        message_out = ("Node:",str(my_rank),"finished","","\r\n")
        print ("Message out for slaves:",message_out)
        comm.send(message_out,dest=0)
        
    MPI.Finalize()        
   
if __name__ == '__main__':
     main()

