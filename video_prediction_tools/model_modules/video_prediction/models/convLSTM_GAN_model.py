# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong,Yanji"
__date__ = "2021-04-13"

import tensorflow as tf
from model_modules.video_prediction.models.model_helpers import set_and_check_pred_frames
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.layers.layer_def import batch_norm
from model_modules.video_prediction.models.vanilla_convLSTM_model import VanillaConvLstmVideoPredictionModel as convLSTM
from .our_base_model import BaseModels

class ConvLstmGANVideoPredictionModel(BaseModels):

    def __init__(self, hparams_dict_config=None, mode='train'):
        super().__init__(hparams_dict_config)
        self.bd1 = batch_norm(name = "bd1")
        self.bd2 = batch_norm(name = "bd2")
        self.bd3 = batch_norm(name = "dis3")

    def parse_hparams(self, hparams):
        """
        Obtain the hparams from the dict to the class variables
        """
        try:
            self.context_frames = self.hparams.context_frames
            self.max_epochs = self.hparams.max_epochs
            self.batch_size = self.hparams.batch_size
            self.shuffle_on_val = self.hparams.shuffle_on_val
            self.loss_fun = self.hparams.loss_fun
            self.recon_weight = self.hparams.recon_weight
            self.learning_rate = self.hparams.lr
            self.sequence_length = self.hparams.sequence_length
            self.opt_var = self.hparams.opt_var
            self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
            self.ngf = self.hparams.ngf
            self.ndf = self.hparams.ndf

        except Exception as error:
           print("error: {}".format(error))
           raise ValueError("Method %{}: the hyper-parameter dictionary must include parameters above")


    def build_graph(self, x: tf.Tensor):

        self.inputs = x

        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()

        #Build graph
        x_hat = self.build_model(x)

        #Get losses (reconstruciton loss, total loss and descriminator loss)
        self.total_loss = self.get_loss(x, x_hat)

        #Define optimizer
        self.train_op = self.optimizer(self.total_loss)

        #Save to outputs
        self.outputs["gen_images"] = x_hat
        self.outputs["total_loss"] = self.total_loss
        # Summary op
        sum_dict = {"total_loss": self.total_loss,
                  "D_loss": self.D_loss,
                  "G_loss": self.G_loss,
                  "D_loss_fake": self.D_loss_fake,
                  "D_loss_real": self.D_loss_real,
                  "recon_loss": self.recon_loss}

        self.summary_op = self.summary(**sum_dict)
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        self.is_build_graph = True
        return self.is_build_graph

    def get_loss(self, x: tf.Tensor, x_hat: tf.Tensor):
        """
        We use the loss from vanilla convolutional LSTM as reconstruction loss
        """
        self.G_loss = self.get_gen_loss()
        self.D_loss = self.get_disc_loss()
        self._get_vars()
        #self.recon_loss = self.get_loss(self, x, x_hat) #use the loss from vanilla convLSTM

        if self.opt_var == "all":
            x = x[:, self.context_frames:, :, :, :]
            print("The model is optimzied on all the variables in the loss function")
        elif self.opt_var != "all" and isinstance(self.opt_var, str):
            self.opt_var = int(self.opt_var)
            print("The model is optimized on the {} variable in the loss function".format(self.opt_var))
            x = x[:, self.context_frames:, :, :, self.opt_var]
            x_hat = x_hat[:, :, :, :, self.opt_var]
        else:
            raise ValueError("The opt var in the hyperparameters setup should be '0','1','2' indicate the index of target variable to be optimised or 'all' indicating optimize all the variables")

        if self.loss_fun == "mse":
            self.recon_loss = tf.reduce_mean(tf.square(x - x_hat))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(x, [-1])
            x_hat_predict_frames_flatten = tf.reshape(x_hat, [-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            self.recon_loss = bce(x_flatten, x_hat_predict_frames_flatten)
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'mse' or 'cross_entropy'")

        self.D_loss = (1 - self.recon_weight) * self.D_loss
        total_loss = (1-self.recon_weight) * self.G_loss + self.recon_weight*self.recon_loss
        return total_loss

    def optimizer(self, *args):

        if self.mode == "train":
            if self.recon_weight == 1:
                print("Only train generator- ConvLSTM")
                train_op = tf.train.AdamOptimizer(learning_rate =
                                                       self.learning_rate).\
                    minimize(self.total_loss, var_list=self.gen_vars)
            else:
                print("Training discriminator")
                self.D_solver = tf.train.AdamOptimizer(learning_rate =self.learning_rate).\
                    minimize(self.D_loss, var_list=self.disc_vars)
                with tf.control_dependencies([self.D_solver]):
                    print("Training generator....")
                    self.G_solver = tf.train.AdamOptimizer(learning_rate =self.learning_rate).\
                        minimize(self.total_loss, var_list=self.gen_vars)
                with tf.control_dependencies([self.G_solver]):
                    train_op = tf.assign_add(self.global_step, 1)
        else:
           train_op = None
        return train_op


    def build_model(self, x):
        """
        Define gan architectures
        """
        #conditional GAN
        x_hat = self.generator(x)

        self.D_real, self.D_real_logits = self.discriminator(self.inputs[:, self.context_frames:, :, :, 0:1])
        self.D_fake, self.D_fake_logits = self.discriminator(x_hat[:, self.context_frames - 1:, :, :, 0:1])

        return x_hat


    def generator(self,x):
        """
        Function to build up the generator architecture
        args:
            input images: a input tensor with dimension (n_batch,sequence_length,height,width,channel)
        """
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
            network_template = tf.make_template('network',
                                                convLSTM.convLSTM_cell)  # make the template to share the variables

            x_hat = convLSTM.convLSTM_network(self.inputs,
                                              self.sequence_length,
                                              self.context_frames,
                                              network_template)
        return x_hat


    def discriminator(self, vid):
        """
        Function that get discriminator architecture
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv3d(vid,64,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME", name="dis1")
            conv1 = self._lrelu(conv1)
            conv2 = tf.layers.conv3d(conv1, 128, kernel_size=[4,4,4],strides=[2,2,2],padding="SAME", name="dis2")
            conv2 = self._lrelu(self.bd1(conv2))
            conv3 = tf.layers.conv3d(conv2, 256, kernel_size=[4,4,4],strides=[2,2,2],padding="SAME" ,name="dis3")
            conv3 = self._lrelu(self.bd2(conv3))
            conv4 = tf.layers.conv3d(conv3, 512, kernel_size=[4,4,4],strides=[2,2,2],padding="SAME", name="dis4")
            conv4 = self._lrelu(self.bd3(conv4))
            conv5 = tf.layers.conv3d(conv4, 1, kernel_size=[2,4,4],strides=[1,1,1],padding="SAME", name="dis5")
            conv5 = tf.reshape(conv5, [-1,1])
            conv5sigmoid = tf.nn.sigmoid(conv5)
            return conv5sigmoid, conv5

    def get_disc_loss(self):
        """
        Return the loss of discriminator given inputs
        """
        real_labels = tf.ones_like(self.D_real)
        gen_labels = tf.zeros_like(self.D_fake)
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=real_labels))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=gen_labels))
        D_loss = self.D_loss_real + self.D_loss_fake
        return D_loss



    def get_gen_loss(self):
        """
        Param:
	    num_images    : the number of images the generator should produce, which is also the lenght of the real image
            z_dim     : the dimension of the noise vector, a scalar
        Return the loss of generator given inputs
        """
        real_labels = tf.ones_like(self.D_fake)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,
                                                                             labels=real_labels))
        return G_loss
   
    def _get_vars(self):
        """
        Get trainable variables from discriminator and generator
        """
        self.disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        self.gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]



    def _lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    def _linear(self, input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev = stddev))
            bias = tf.get_variable("bias", [output_size],
                                   initializer = tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias



