__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-11-05"

import collections
import functools
import itertools
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from model_modules.video_prediction import ops, flow_ops
from model_modules.video_prediction.models import BaseVideoPredictionModel
from model_modules.video_prediction.models import networks
from model_modules.video_prediction.ops import dense, pad2d, conv2d, flatten, tile_concat
from model_modules.video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from model_modules.video_prediction.utils import tf_utils
from datetime import datetime
from pathlib import Path
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.layers.BasicConvLSTMCell import BasicConvLSTMCell
from tensorflow.contrib.training import HParams

class VanillaConvLstmVideoPredictionModel(object):
    def __init__(self, mode='train', hparams_dict=None):
        """
        This is class for building convLSTM architecture by using updated hparameters
        args:
             mode   :str, "train" or "val", side note: mode may not be used in the convLSTM, but this will be a useful argument for the GAN-based model
             hparams_dict: dict, the dictionary contains the hparaemters names and values
        """
        self.mode = mode
        self.hparams_dict = hparams_dict
        self.hparams = self.parse_hparams()        
        self.learning_rate = self.hparams.lr
        self.total_loss = None
        self.context_frames = self.hparams.context_frames
        self.sequence_length = self.hparams.sequence_length
        self.predict_frames = self.sequence_length - self.context_frames
        self.max_epochs = self.hparams.max_epochs
        self.loss_fun = self.hparams.loss_fun


    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self):
        """
        Parse the hparams setting to ovoerride the default ones
        """
        
        parsed_hparams = self.get_default_hparams().override_from_dict(self.hparams_dict or {})
        return parsed_hparams


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
            loss_fun = "cross_entropy",
            shuffle_on_val= True,
         )
        return hparams


    def build_graph(self, x):
        self.is_build_graph = False
        self.x = x["images"]
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        # ARCHITECTURE
        self.gan_network()
        #This is the loss function (RMSE):
        #This is loss function only for 1 channel (temperature RMSE)
        if self.loss_fun == "rmse":
            self.total_loss = tf.reduce_mean(
                tf.square(self.x[:, self.context_frames:,:,:,0] - self.x_hat_predict_frames[:,:,:,:,0]))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(self.x[:, self.context_frames:,:,:,0],[-1])
            x_hat_predict_frames_flatten = tf.reshape(self.x_hat_predict_frames[:,:,:,:,0],[-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            self.total_loss = bce(x_flatten,x_hat_predict_frames_flatten)  
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'rmse' or 'cross_entropy'")

        #This is the loss for only all the channels(temperature, geo500, pressure)
        #self.total_loss = tf.reduce_mean(
        #    tf.square(self.x[:, self.context_frames:,:,:,:] - self.x_hat_predict_frames[:,:,:,:,:]))            
 
        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(self.total_loss, global_step = self.global_step)
        self.outputs = {}
        self.outputs["gen_images"] = self.x_hat
        # Summary op
        self.loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        self.is_build_graph = True
        return self.is_build_graph 

    
    def gan_network(self):
        pass




   def get_noise(n_samples,z_dim):
       """
       Function for creating noise: Given the dimensions (n_samples,z_dim)
       """ 
       return np.random.uniform(-1., 1., size=[n_samples, z_dim])



   def get_generator_block(input_dim,output_dim,idx):
       output1 = ld.conv_layer(input_dim,2,stride=1,num_features=output_dim,idx=idx,activate="linear")
       output2 = ld.bn_layers(output1,idx,is_train=False)
       output3 = tf.nn.relu(output2)
       return output_3


   def generator(z_dim,im_dim,hidden_dim):
       """

       """

   
