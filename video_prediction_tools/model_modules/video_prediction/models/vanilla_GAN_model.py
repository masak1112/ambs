__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2021=01-05"

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
        # Architecture
        self.gan_network()
        #This is the loss function (RMSE):
        #This is loss function only for 1 channel (temperature RMSE)
        D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=theta_G)
        self.total_loss = bce(x_flatten,x_hat_predict_frames_flatten)  

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

    

   def get_noise(self,n_samples,z_dim):
       """
       Function for creating noise: Given the dimensions (n_samples,z_dim)
       """ 
       return np.random.uniform(-1., 1., size=[n_samples, z_dim])


   def get_generator_block(self,inputs,output_dim,idx):
       
       """
       Generator Block
       Function for return a neural network of the generator given input and output dimensions
       args:
            inputs : the  input vector
            output_dim: the dimeniosn of output vector
       return:
             a generator neural network layer, with a convolutional layers followed by batch normalization and a relu activation
       
       """
       output1 = ld.conv_layer(inputs,kernel_size=2,stride=1,num_features=output_dim,idx=idx,activate="linear")
       output2 = ld.bn_layers(output1,idx,is_train=False)
       output3 = tf.nn.relu(output2)
       return output_3


   def generator(self,noise,im_dim,hidden_dim):
       """
       Function to build up the generator architecture
       args:
           noise: a noise tensor with dimension (n_samples,z_dim)
           im_dim: the dimension of the input image
           hidden_dim: the inner dimension
       """
       with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
           layer1 = self.get_generator_block(noise,hidden_dim,1)
           layer2 = self.get_generator_block(layer1,hidden_dim*2,2)
           layer3 = self.get_generator_block(layer2,hidden_dim*4,3)
           layer4 = self.get_generator_block(layer3,hidden_dim*8,4)
           layer5 = ld.conv_layer(layer4,kernel_size=2,stride=1,num_features=im_dim,idx=5,activate="linear")
           layer6 = tf.nn.sigmoid(layer5,name="6_conv")
       return layer6



   def get_discriminator_block(self,inputs,output_dim,idx):

       """
       Distriminator block
       Function for ruturn a neural network of a descriminator given input and output dimensions

       args:
           inputs : the dimension of input vector
           output_dim: the dimension of output dim
           idx:      : the index for the namespace of this block
       Return:
           a distriminator neural network layer with a convolutional layers followed by a leakyRelu function 
       """
       output1 = ld.conv_layer(inputs,2,stride=1,nun_features=output_dim,idx=idx,activate="linear")
       output2 = tf.nn.leaky_relu(output1)
       return output2


   def discriminator(self,image,hidden_dim):
       """
       Function that get discriminator architecture      
       """
       with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
           layer1 = self.get_discriminator_block(image,hidden_dim)
           layer2 = self.get_discriminator_block(layer1,hidden_dim*4)
           layer3 = self.get_discriminator_block(layer2,hidden_dim*2)
           layer4 = self.get_discriminator_block(layer3,hidden_dim)
           layer5 = tf.nn.sigmoid(layer4,1)
       return layer5


   def get_disc_loss(self):
       """
       Return the loss of discriminator given inputs
       """
       noise = self.get_noise(1000,10)
       G_samples = self.generator(noise)
       D_real = self.discriminator(image)
       D_fake = self.discriminator(G_samples)
       real_labels = tf.ones_like(D_real)
       gen_labels = tf.zeros_like(D_fake)
       D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=real_labels))
       D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=get_labels))
       self.D_loss = D_loss_real + D_loss_fake
       return self.D_loss


   def get_gen_loss(self,num_images,z_dim):
       """
       Param:
	    num_images: the number of images the generator should produce, which is also the lenght of the real image
            z_dim     : the dimension of the noise vector, a scalar
       Return the loss of generator given inputs
       """
       noises = self.get_noise(num_images,z_dim)
       gen_images = self.generator(noise,im_dim,hidden_dim)
       disc_gen_images = self.disrciminator(gen_images,hidden_dim)
       real_labels = tf.ones_like(gen_images)
       self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_gen_images, labels=real_labels))
       return self.gen_loss         
   
   def get_vars(self):
       """
       Get trainable variables from discriminator and generator
       """
       self.disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("disc")]
       self.gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("gen")]

 
  
   def define_gan(self,image):
       """
       Define gan architectures
       """
       noise = self.get_noise(1000,10)
       G_samples = self.generator(noise)
       D_real = self.discriminator(image)
       D_fake = self.discriminator(G_samples)
       discriminator.trainable = False
                  

   
