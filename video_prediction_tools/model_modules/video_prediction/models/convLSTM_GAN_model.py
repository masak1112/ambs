__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong,Yanji"
__date__ = "2021-04-13"

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
from .vanilla_convLSTM_model import VanillaConvLstmVideoPredictionModel

class ConvLstmGANVideoPredictionModel(object):
    def __init__(self, mode='train', hparams_dict=None):
        """
        This is class for building convLSTM_GAN architecture by using updated hparameters
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
        self.batch_size = self.hparams.batch_size
        self.recon_weight = self.hparams.recon_weight
       
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
            recon_wegiht    : the weight for reconstrution loss
            """
        hparams = dict(
            context_frames=12,
            sequence_length=24,
            max_epochs = 20,
            batch_size = 40,
            lr = 0.001,
            loss_fun = "cross_entropy",
            shuffle_on_val= True,
            recon_weight=0.99,
          
         )
        return hparams


    def build_graph(self, x):
        self.is_build_graph = False
        self.x = x["images"]
        self.width = self.x.shape.as_list()[3]
        self.height = self.x.shape.as_list()[2]
        self.channels = self.x.shape.as_list()[4]
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        # Architecture
        self.define_gan()
        #This is the loss function (RMSE):
        #This is loss function only for 1 channel (temperature RMSE)
        #generator los
        self.total_loss = (1-self.recon_weight) * self.G_loss + self.recon_weight*self.recon_loss
        if self.mode == "train":
            print("Training distriminator")
            self.D_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.D_loss, var_list=self.disc_vars)
            with tf.control_dependencies([self.D_solver]):
                print("Training generator....")
                self.G_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.total_loss, var_list=self.gen_vars)
            with tf.control_dependencies([self.G_solver]):
                self.train_op = tf.assign_add(self.global_step,1)
        else:
           self.train_op = None 

        self.outputs = {}
        self.outputs["gen_images"] = self.gen_images
        self.outputs["total_loss"] = self.total_loss
        # Summary op
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("D_loss", self.D_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        tf.summary.scalar("D_loss_fake", self.D_loss_fake) 
        tf.summary.scalar("D_loss_real", self.D_loss_real)
        tf.summary.scalar("recon_loss",self.recon_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        self.is_build_graph = True
        return self.is_build_graph 
    
    def get_noise(self):
        """
        Function for creating noise: Given the dimensions (n_batch,n_seq, n_height, n_width, channel)
        """ 
        self.noise = tf.random.uniform(minval=-1., maxval=1., shape=[self.batch_size, self.sequence_length, self.height, self.width, self.channels])
        return self.noise



    def generator(self):
        """
        Function to build up the generator architecture
        args:
            noise: a noise tensor with dimension (n_batch,sequence_length,height,width,channel)
        """
        with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
            layer_gen = self.convLSTM_network(self.noise)
        return layer_gen


    def discriminator(self,image):
        """
        Function that get discriminator architecture      
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            layer_disc = self.convLSTM_network(image)
        return layer_disc


    def get_disc_loss(self):
        """
        Return the loss of discriminator given inputs
        """
          
        real_labels = tf.ones_like(self.D_real)
        gen_labels = tf.zeros_like(self.D_fake)
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=real_labels))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=gen_labels))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        return self.D_loss


    def get_gen_loss(self):
        """
        Param:
	    num_images: the number of images the generator should produce, which is also the lenght of the real image
            z_dim     : the dimension of the noise vector, a scalar
        Return the loss of generator given inputs
        """
        real_labels = tf.ones_like(self.gen_images)
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=real_labels))
        return self.G_loss         
   
    def get_vars(self):
        """
        Get trainable variables from discriminator and generator
        """
        self.disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        self.gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
       
 
  
    def define_gan(self):
        """
        Define gan architectures
        """
        self.noise = self.get_noise()
        self.gen_images = self.generator()
        self.D_real = self.discriminator(self.x)
        self.D_fake = self.discriminator(self.gen_images)
        self.get_gen_loss()
        self.get_disc_loss()
        self.get_vars()
        if self.loss_fun == "rmse":
            self.recon_loss = tf.reduce_mean(tf.square(self.x[:, self.context_frames:,:,:,0] - self.gen_images[:,:,:,:,0]))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(self.x[:, self.context_frames:,:,:,0],[-1])
            x_hat_predict_frames_flatten = tf.reshape(self.gen_images[:,:,:,:,0],[-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            self.recon_loss = bce(x_flatten,x_hat_predict_frames_flatten)
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'rmse' or 'cross_entropy'")   


    @staticmethod
    def convLSTM_cell(inputs, hidden):
        y_0 = inputs #we only usd patch 1, but the original paper use patch 4 for the moving mnist case, but use 2 for Radar Echo Dataset
        channels = inputs.get_shape()[-1]
        # conv lstm cell
        cell_shape = y_0.get_shape().as_list()
        channels = cell_shape[-1]
        with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell(shape = [cell_shape[1], cell_shape[2]], filter_size=5, num_features=64)
            if hidden is None:
                hidden = cell.zero_state(y_0, tf.float32)
            output, hidden = cell(y_0, hidden)
        output_shape = output.get_shape().as_list()
        z3 = tf.reshape(output, [-1, output_shape[1], output_shape[2], output_shape[3]])
        #we feed the learn representation into a 1 × 1 convolutional layer to generate the final prediction
        x_hat = ld.conv_layer(z3, 1, 1, channels, "decode_1", activate="sigmoid")
        return x_hat, hidden

    def convLSTM_network(self,x):
        network_template = tf.make_template('network',VanillaConvLstmVideoPredictionModel.convLSTM_cell)  # make the template to share the variables
        # create network
        x_hat = []
        
        #This is for training (optimization of convLSTM layer)
        hidden_g = None
        for i in range(self.sequence_length-1):
            if i < self.context_frames:
                x_1_g, hidden_g = network_template(x[:, i, :, :, :], hidden_g)
            else:
                x_1_g, hidden_g = network_template(x_1_g, hidden_g)
            x_hat.append(x_1_g)

        # pack them all together
        x_hat = tf.stack(x_hat)
        self.x_hat= tf.transpose(x_hat, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat_predict_frames = self.x_hat[:,self.context_frames-1:,:,:,:]
        return self.x_hat_predict_frames
     


   
