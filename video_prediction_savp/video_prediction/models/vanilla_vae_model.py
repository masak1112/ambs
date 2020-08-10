import collections
import functools
import itertools
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from video_prediction import ops, flow_ops
from video_prediction.models import BaseVideoPredictionModel
from video_prediction.models import networks
from video_prediction.ops import dense, pad2d, conv2d, flatten, tile_concat
from video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from video_prediction.utils import tf_utils
from datetime import datetime
from pathlib import Path
from video_prediction.layers import layer_def as ld

class VanillaVAEVideoPredictionModel(BaseVideoPredictionModel):
    def __init__(self, mode='train', aggregate_nccl=None,hparams_dict=None,
                 hparams=None,**kwargs):
        super(VanillaVAEVideoPredictionModel, self).__init__(mode, hparams_dict, hparams, **kwargs)
        self.mode = mode
        self.learning_rate = self.hparams.lr
        self.weight_recon = self.hparams.weight_recon
        self.nz = self.hparams.nz
        self.aggregate_nccl=aggregate_nccl
        self.gen_images_enc = None
        self.train_op = None
        self.summary_op = None
        self.recon_loss = None
        self.latent_loss = None
        self.total_loss = None
        

    def get_default_hparams_dict(self):
        """
        The keys of this dict define valid hyperparameters for instances of
        this class. A class inheriting from this one should override this
        method if it has a different set of hyperparameters.

        Returns:
            A dict with the following hyperparameters.

            batch_size: batch size for training.
            lr: learning rate. if decay steps is non-zero, this is the
                learning rate for steps <= decay_step.
            end_lr: learning rate for steps >= end_decay_step if decay_steps
                is non-zero, ignored otherwise.
            decay_steps: (decay_step, end_decay_step) tuple.
            max_epochs: number of training epochs.
            beta1: momentum term of Adam.
            beta2: momentum term of Adam.
            context_frames: the number of ground-truth frames to pass in at
                start. Must be specified during instantiation.
            sequence_length: the number of frames in the video sequence,
                including the context frames, so this model predicts
                `sequence_length - context_frames` future frames. Must be
                specified during instantiation.
        """
        default_hparams = super(VanillaVAEVideoPredictionModel, self).get_default_hparams_dict()
        print ("default hparams",default_hparams)
        hparams = dict(
            batch_size=16,
            lr=0.001,
            end_lr=0.0,
            decay_steps=(200000, 300000),
            lr_boundaries=(0,),
            max_epochs=35,
            nz=10,
            context_frames=-1,
            sequence_length=-1,
            weight_recon = 0.4        
            
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))


    def build_graph(self,x):  
        self.x = x["images"]
        #self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        self.x_hat, self.z_log_sigma_sq, self.z_mu = self.vae_arc_all()
        self.recon_loss = tf.reduce_mean(tf.square(self.x[:, 1:, :, :, 0] - self.x_hat[:, :-1, :, :, 0]))
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) -
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)
        self.total_loss = self.weight_recon * self.recon_loss + self.latent_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(self.total_loss, global_step=self.global_step)
        # Build a saver

        self.losses = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_loss,
        }

        # Summary op
        self.loss_summary = tf.summary.scalar("recon_loss", self.recon_loss)
        self.loss_summary = tf.summary.scalar("latent_loss", self.latent_loss)
        self.loss_summary = tf.summary.scalar("total_loss", self.latent_loss)
        self.summary_op = tf.summary.merge_all()

        self.outputs = {}
        self.outputs["gen_images"] = self.x_hat
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        return None


    @staticmethod
    def vae_arc3(x,l_name=0,nz=16):
        seq_name = "sq_" + str(l_name) + "_"
        conv1 = ld.conv_layer(x, 3, 2, 8, seq_name + "encode_1")
        conv2 = ld.conv_layer(conv1, 3, 1, 8, seq_name + "encode_2")
        conv3 = ld.conv_layer(conv2, 3, 2, 8, seq_name + "encode_3")
        conv4 = tf.layers.Flatten()(conv3)
        conv3_shape = conv3.get_shape().as_list()
        z_mu = ld.fc_layer(conv4, hiddens = nz, idx = seq_name + "enc_fc4_m")
        z_log_sigma_sq = ld.fc_layer(conv4, hiddens = nz, idx = seq_name + "enc_fc4_m"'enc_fc4_sigma')
        eps = tf.random_normal(shape = tf.shape(z_log_sigma_sq), mean = 0, stddev = 1, dtype = tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps        
        z2 = ld.fc_layer(z, hiddens = conv3_shape[1] * conv3_shape[2] * conv3_shape[3], idx = seq_name + "decode_fc1") 
        z3 = tf.reshape(z2, [-1, conv3_shape[1], conv3_shape[2], conv3_shape[3]])
        conv5 = ld.transpose_conv_layer(z3, 3, 2, 8, seq_name + "decode_5")  
        conv6  = ld.transpose_conv_layer(conv5, 3, 1, 8,seq_name + "decode_6")
        x_hat = ld.transpose_conv_layer(conv6, 3, 2, 3, seq_name + "decode_8")
        return x_hat, z_mu, z_log_sigma_sq, z

    def vae_arc_all(self):
        X = []
        z_log_sigma_sq_all = []
        z_mu_all = []
        for i in range(20):
            q, z_mu, z_log_sigma_sq, z = VanillaVAEVideoPredictionModel.vae_arc3(self.x[:, i, :, :, :], l_name=i, nz=self.nz)
            X.append(q)
            z_log_sigma_sq_all.append(z_log_sigma_sq)
            z_mu_all.append(z_mu)
        x_hat = tf.stack(X, axis = 1)
        z_log_sigma_sq_all = tf.stack(z_log_sigma_sq_all, axis = 1)
        z_mu_all = tf.stack(z_mu_all, axis = 1)

        return x_hat, z_log_sigma_sq_all, z_mu_all
