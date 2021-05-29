__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2021=01-05"



"""
This code implement take the following as references:
1) https://stackabuse.com/introduction-to-gans-with-python-and-tensorflow/
2) cousera GAN courses
3) https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
"""

import tensorflow as tf

from model_modules.video_prediction.models.model_helpers import set_and_check_pred_frames
from model_modules.video_prediction.layers import layer_def as ld
from tensorflow.contrib.training import HParams

class VanillaGANVideoPredictionModel(object):
    def __init__(self, mode='train', hparams_dict=None):
        """
        This is class for building vanilla GAN architecture by using updated hparameters
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
        self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
        self.max_epochs = self.hparams.max_epochs
        self.loss_fun = self.hparams.loss_fun
        self.batch_size = self.hparams.batch_size
        self.z_dim = self.hparams.z_dim  # dim of noise-vector

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
            context_frames=12,
            sequence_length=24,
            max_epochs = 20,
            batch_size = 40,
            lr = 0.001,
            loss_fun = "cross_entropy",
            shuffle_on_val= True,
            z_dim = 32,
         )
        return hparams


    def build_graph(self, x):
        self.is_build_graph = False
        self.x = x["images"]
        self.width = self.x.shape.as_list()[3]
        self.height = self.x.shape.as_list()[2]
        self.channels = self.x.shape.as_list()[4]
        self.n_samples = self.x.shape.as_list()[0] * self.x.shape.as_list()[1]
        self.x = tf.reshape(self.x, [-1, self.height,self.width,self.channels]) 
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        # Architecture
        self.define_gan()
        #This is the loss function (RMSE):
        #This is loss function only for 1 channel (temperature RMSE)
        if self.mode == "train":
            self.D_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.D_loss, var_list=self.disc_vars)
            with tf.control_dependencies([self.D_solver]):
                self.G_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.G_loss, var_list=self.gen_vars)
            with tf.control_dependencies([self.G_solver]):
                self.train_op = tf.assign_add(self.global_step,1)
        else:
           self.train_op = None 
        self.total_loss = self.G_loss + self.D_loss 
        self.outputs = {}
        self.outputs["gen_images"] = self.gen_images
        self.outputs["total_loss"] = self.total_loss
        # Summary op
        self.loss_summary = tf.summary.scalar("total_loss", self.G_loss + self.D_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        self.is_build_graph = True
        return self.is_build_graph 
    
    def get_noise(self):
        """
        Function for creating noise: Given the dimensions (n_samples,z_dim)
        """ 
        self.noise = tf.random.uniform(minval=-1., maxval=1., shape=[self.n_samples, self.height, self.width, self.channels])
        return self.noise

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
        output2 = ld.bn_layers(output1,idx,is_training=False)
        output3 = tf.nn.relu(output2)
        return output3


    def generator(self,hidden_dim):
        """
        Function to build up the generator architecture
        args:
            noise: a noise tensor with dimension (n_samples,height,width,channel)
            hidden_dim: the inner dimension
        """
        with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
            layer1 = self.get_generator_block(self.noise,hidden_dim,1)
            layer2 = self.get_generator_block(layer1,hidden_dim*2,2)
            layer3 = self.get_generator_block(layer2,hidden_dim*4,3)
            layer4 = self.get_generator_block(layer3,hidden_dim*8,4)
            layer5 = ld.conv_layer(layer4,kernel_size=2,stride=1,num_features=self.channels,idx=5,activate="linear")
            layer6 = tf.nn.sigmoid(layer5,name="6_conv")
        print("layer6",layer6)
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
        output1 = ld.conv_layer(inputs,2,stride=1,num_features=output_dim,idx=idx,activate="linear")
        output2 = tf.nn.leaky_relu(output1)
        return output2


    def discriminator(self,image,hidden_dim):
        """
        Function that get discriminator architecture      
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            layer1 = self.get_discriminator_block(image,hidden_dim,idx=1)
            layer2 = self.get_discriminator_block(layer1,hidden_dim*4,idx=2)
            layer3 = self.get_discriminator_block(layer2,hidden_dim*2,idx=3)
            layer4 = self.get_discriminator_block(layer3, self.channels,idx=4)
            layer5 = tf.nn.sigmoid(layer4)
        return layer5


    def get_disc_loss(self):
        """
        Return the loss of discriminator given inputs
        """
          
        real_labels = tf.ones_like(self.D_real)
        gen_labels = tf.zeros_like(self.D_fake)
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=real_labels))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=gen_labels))
        self.D_loss = D_loss_real + D_loss_fake
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
        self.gen_images = self.generator(hidden_dim=8)
        self.D_real = self.discriminator(self.x,hidden_dim=8)
        self.D_fake = self.discriminator(self.gen_images,hidden_dim=8)
        self.get_gen_loss()
        self.get_disc_loss()
        self.get_vars()
      
