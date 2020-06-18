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
from video_prediction.layers.BasicConvLSTMCell import BasicConvLSTMCell
from video_prediction.layers.mcnet_ops import *
from video_prediction.utils.mcnet_utils import *
import os

class McNetVideoPredictionModel(BaseVideoPredictionModel):
    def __init__(self, mode='train', hparams_dict=None,
                 hparams=None, **kwargs):
        super(McNetVideoPredictionModel, self).__init__(mode, hparams_dict, hparams, **kwargs)
        self.mode = mode
        self.lr = self.hparams.lr
        self.context_frames = self.hparams.context_frames
        self.sequence_length = self.hparams.sequence_length
        self.predict_frames = self.sequence_length - self.context_frames
        self.df_dim = self.hparams.df_dim
        self.gf_dim = self.hparams.gf_dim
        self.alpha = self.hparams.alpha
        self.beta = self.hparams.beta
        self.gen_images_enc = None
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




            max_steps: number of training steps.


            context_frames: the number of ground-truth frames to pass in at
                start. Must be specified during instantiation.
            sequence_length: the number of frames in the video sequence,
                including the context frames, so this model predicts
                `sequence_length - context_frames` future frames. Must be
                specified during instantiation.
            df_dim: specific parameters for mcnet
            gf_dim: specific parameters for menet
            alpha:  specific parameters for mcnet
            beta:   specific paramters for mcnet

        """
        default_hparams = super(McNetVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            batch_size=16,
            lr=0.001,
            max_steps=350000,
            context_frames = 10,
            sequence_length = 20,
            nz = 16,
            gf_dim = 64,
            df_dim = 64,
            alpha = 1,
            beta = 0.0
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def build_graph(self, x):

        self.x = x["images"]
        self.x_shape = self.x.get_shape().as_list()
        self.batch_size = self.x_shape[0]
        self.image_size = [self.x_shape[2],self.x_shape[3]]
        self.c_dim = self.x_shape[4]
        self.diff_shape = [self.batch_size, self.context_frames-1, self.image_size[0],
                           self.image_size[1], self.c_dim]
        self.xt_shape = [self.batch_size, self.image_size[0], self.image_size[1],self.c_dim]
        self.is_train = True
       

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        original_global_variables = tf.global_variables()

        # self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
        self.xt = self.x[:, self.context_frames - 1, :, :, :]

        self.diff_in = tf.placeholder(tf.float32, self.diff_shape, name='diff_in')
        diff_in_all = []
        for t in range(1, self.context_frames):
            prev = self.x[:, t-1:t, :, :, :]
            next = self.x[:, t:t+1, :, :, :]
            #diff_in = tf.reshape(next - prev, [self.batch_size, 1, self.image_size[0], self.image_size[1], -1])
            print("prev:",prev)
            print("next:",next)
            diff_in = tf.subtract(next,prev)
            print("diff_in:",diff_in)
            diff_in_all.append(diff_in)

        self.diff_in = tf.concat(axis = 1, values = diff_in_all)

        cell = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8], [3, 3], 256)

        pred = self.forward(self.diff_in, self.xt, cell)


        self.G = tf.concat(axis=1, values=pred)#[batch_size,context_frames,image1,image2,channels]
        print ("1:self.G:",self.G)
        if self.is_train:

            true_sim = self.x[:, self.context_frames:, :, :, :]

            # Bing: the following make sure the channel is three dimension, if the channle is 3 then will be duplicated
            if self.c_dim == 1: true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])

            # Bing: the raw inputs shape is [batch_size, image_size[0],self.image_size[1], num_seq, channel]. tf.transpose will transpoe the shape into
            # [batch size*num_seq, image_size0, image_size1, channels], for our era5 case, we do not need transpose
            # true_sim = tf.reshape(tf.transpose(true_sim,[0,3,1,2,4]),
            #                             [-1, self.image_size[0],
            #                              self.image_size[1], 3])
            true_sim = tf.reshape(true_sim, [-1, self.image_size[0], self.image_size[1], 3])




        gen_sim = self.G
        
        #combine groud truth and predict frames
        self.x_hat = tf.concat([self.x[:, :self.context_frames, :, :, :], self.G], 1)
        print ("self.x_hat:",self.x_hat)
        if self.c_dim == 1: gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
        # gen_sim = tf.reshape(tf.transpose(gen_sim,[0,3,1,2,4]),
        #                                [-1, self.image_size[0],
        #                                self.image_size[1], 3])

        gen_sim = tf.reshape(gen_sim, [-1, self.image_size[0], self.image_size[1], 3])


        binput = tf.reshape(tf.transpose(self.x[:, :self.context_frames, :, :, :], [0, 1, 2, 3, 4]),
                            [self.batch_size, self.image_size[0],
                             self.image_size[1], -1])

        btarget = tf.reshape(tf.transpose(self.x[:, self.context_frames:, :, :, :], [0, 1, 2, 3, 4]),
                             [self.batch_size, self.image_size[0],
                              self.image_size[1], -1])
        bgen = tf.reshape(self.G, [self.batch_size,
                                   self.image_size[0],
                                   self.image_size[1], -1])

        print ("binput:",binput)
        print("btarget:",btarget)
        print("bgen:",bgen)

        good_data = tf.concat(axis=3, values=[binput, btarget])
        gen_data = tf.concat(axis=3, values=[binput, bgen])
        self.gen_data = gen_data
        print ("2:self.gen_data:", self.gen_data)
        with tf.variable_scope("DIS", reuse=False):
            self.D, self.D_logits = self.discriminator(good_data)

        with tf.variable_scope("DIS", reuse=True):
            self.D_, self.D_logits_ = self.discriminator(gen_data)

        self.L_p = tf.reduce_mean(
            tf.square(self.G - self.x[:, self.context_frames:, :, :, :]))

        self.L_gdl = gdl(gen_sim, true_sim, 1.)
        self.L_img = self.L_p + self.L_gdl

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits, labels = tf.ones_like(self.D)
            ))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_, labels = tf.zeros_like(self.D_)
            ))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.L_GAN = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_, labels = tf.ones_like(self.D_)
            ))

        self.loss_sum = tf.summary.scalar("L_img", self.L_img)
        self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
        self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
        self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.total_loss = self.alpha * self.L_img + self.beta * self.L_GAN
        self._loss_sum = tf.summary.scalar("total_loss", self.total_loss)
        self.g_sum = tf.summary.merge([self.L_p_sum,
                                       self.L_gdl_sum, self.loss_sum,
                                       self.L_GAN_sum])
        self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum,
                                       self.d_loss_fake_sum])


        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
        self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
        num_param = 0.0
        for var in self.g_vars:
            num_param += int(np.prod(var.get_shape()));
        print("Number of parameters: %d" % num_param)

        # Training
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(
            self.d_loss, var_list = self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(
            self.alpha * self.L_img + self.beta * self.L_GAN, var_list = self.g_vars, global_step=self.global_step)
       
        self.train_op = [self.d_optim,self.g_optim]
        self.outputs = {}
        self.outputs["gen_images"] = self.x_hat
        

        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        return 


    def forward(self, diff_in, xt, cell):
        # Initial state
        state = tf.zeros([self.batch_size, self.image_size[0] / 8,
                          self.image_size[1] / 8, 512])
        reuse = False
        # Encoder
        for t in range(self.context_frames - 1):
            enc_h, res_m = self.motion_enc(diff_in[:, t, :, :, :], reuse = reuse)
            h_dyn, state = cell(enc_h, state, scope = 'lstm', reuse = reuse)
            reuse = True
        pred = []
        # Decoder
        for t in range(self.predict_frames):
            if t == 0:
                h_cont, res_c = self.content_enc(xt, reuse = False)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse = False)
                res_connect = self.residual(res_m, res_c, reuse = False)
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse = False)

            else:

                enc_h, res_m = self.motion_enc(diff_in, reuse = True)
                h_dyn, state = cell(enc_h, state, scope = 'lstm', reuse = True)
                h_cont, res_c = self.content_enc(xt, reuse = reuse)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse = True)
                res_connect = self.residual(res_m, res_c, reuse = True)
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse = True)
                print ("x_hat :",x_hat)
            if self.c_dim == 3:
                # Network outputs are BGR so they need to be reversed to use
                # rgb_to_grayscale
                #x_hat_gray = tf.concat(axis=3,values=[x_hat[:,:,:,2:3], x_hat[:,:,:,1:2],x_hat[:,:,:,0:1]])
                #xt_gray = tf.concat(axis=3,values=[xt[:,:,:,2:3], xt[:,:,:,1:2],xt[:,:,:,0:1]])

                #                 x_hat_gray = 1./255.*tf.image.rgb_to_grayscale(
                #                     inverse_transform(x_hat_rgb)*255.
                #                 )
                #                 xt_gray = 1./255.*tf.image.rgb_to_grayscale(
                #                     inverse_transform(xt_rgb)*255.
                #                 )

                x_hat_gray = x_hat
                xt_gray = xt
            else:
                x_hat_gray = inverse_transform(x_hat)
                xt_gray = inverse_transform(xt)

            diff_in = x_hat_gray - xt_gray
            xt = x_hat


            pred.append(tf.reshape(x_hat, [self.batch_size, 1, self.image_size[0],
                                           self.image_size[1], self.c_dim]))

        return pred

    def motion_enc(self, diff_in, reuse):
        res_in = []

        conv1 = relu(conv2d(diff_in, output_dim = self.gf_dim, k_h = 5, k_w = 5,
                            d_h = 1, d_w = 1, name = 'dyn1_conv1', reuse = reuse))
        res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2, 2])

        conv2 = relu(conv2d(pool1, output_dim = self.gf_dim * 2, k_h = 5, k_w = 5,
                            d_h = 1, d_w = 1, name = 'dyn_conv2', reuse = reuse))
        res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2, 2])

        conv3 = relu(conv2d(pool2, output_dim = self.gf_dim * 4, k_h = 7, k_w = 7,
                            d_h = 1, d_w = 1, name = 'dyn_conv3', reuse = reuse))
        res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2, 2])
        return pool3, res_in

    def content_enc(self, xt, reuse):
        res_in = []
        conv1_1 = relu(conv2d(xt, output_dim = self.gf_dim, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv1_1', reuse = reuse))
        conv1_2 = relu(conv2d(conv1_1, output_dim = self.gf_dim, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv1_2', reuse = reuse))
        res_in.append(conv1_2)
        pool1 = MaxPooling(conv1_2, [2, 2])

        conv2_1 = relu(conv2d(pool1, output_dim = self.gf_dim * 2, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv2_1', reuse = reuse))
        conv2_2 = relu(conv2d(conv2_1, output_dim = self.gf_dim * 2, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv2_2', reuse = reuse))
        res_in.append(conv2_2)
        pool2 = MaxPooling(conv2_2, [2, 2])

        conv3_1 = relu(conv2d(pool2, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv3_1', reuse = reuse))
        conv3_2 = relu(conv2d(conv3_1, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv3_2', reuse = reuse))
        conv3_3 = relu(conv2d(conv3_2, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                              d_h = 1, d_w = 1, name = 'cont_conv3_3', reuse = reuse))
        res_in.append(conv3_3)
        pool3 = MaxPooling(conv3_3, [2, 2])
        return pool3, res_in

    def comb_layers(self, h_dyn, h_cont, reuse=False):
        comb1 = relu(conv2d(tf.concat(axis = 3, values = [h_dyn, h_cont]),
                            output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                            d_h = 1, d_w = 1, name = 'comb1', reuse = reuse))
        comb2 = relu(conv2d(comb1, output_dim = self.gf_dim * 2, k_h = 3, k_w = 3,
                            d_h = 1, d_w = 1, name = 'comb2', reuse = reuse))
        h_comb = relu(conv2d(comb2, output_dim = self.gf_dim * 4, k_h = 3, k_w = 3,
                             d_h = 1, d_w = 1, name = 'h_comb', reuse = reuse))
        return h_comb

    def residual(self, input_dyn, input_cont, reuse=False):
        n_layers = len(input_dyn)
        res_out = []
        for l in range(n_layers):
            input_ = tf.concat(axis = 3, values = [input_dyn[l], input_cont[l]])
            out_dim = input_cont[l].get_shape()[3]
            res1 = relu(conv2d(input_, output_dim = out_dim,
                               k_h = 3, k_w = 3, d_h = 1, d_w = 1,
                               name = 'res' + str(l) + '_1', reuse = reuse))
            res2 = conv2d(res1, output_dim = out_dim, k_h = 3, k_w = 3,
                          d_h = 1, d_w = 1, name = 'res' + str(l) + '_2', reuse = reuse)
            res_out.append(res2)
        return res_out

    def dec_cnn(self, h_comb, res_connect, reuse=False):

        shapel3 = [self.batch_size, int(self.image_size[0] / 4),
                   int(self.image_size[1] / 4), self.gf_dim * 4]
        shapeout3 = [self.batch_size, int(self.image_size[0] / 4),
                     int(self.image_size[1] / 4), self.gf_dim * 2]
        depool3 = FixedUnPooling(h_comb, [2, 2])
        deconv3_3 = relu(deconv2d(relu(tf.add(depool3, res_connect[2])),
                                  output_shape = shapel3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv3_3', reuse = reuse))
        deconv3_2 = relu(deconv2d(deconv3_3, output_shape = shapel3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv3_2', reuse = reuse))
        deconv3_1 = relu(deconv2d(deconv3_2, output_shape = shapeout3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv3_1', reuse = reuse))

        shapel2 = [self.batch_size, int(self.image_size[0] / 2),
                   int(self.image_size[1] / 2), self.gf_dim * 2]
        shapeout3 = [self.batch_size, int(self.image_size[0] / 2),
                     int(self.image_size[1] / 2), self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        deconv2_2 = relu(deconv2d(relu(tf.add(depool2, res_connect[1])),
                                  output_shape = shapel2, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv2_2', reuse = reuse))
        deconv2_1 = relu(deconv2d(deconv2_2, output_shape = shapeout3, k_h = 3, k_w = 3,
                                  d_h = 1, d_w = 1, name = 'dec_deconv2_1', reuse = reuse))

        shapel1 = [self.batch_size, self.image_size[0],
                   self.image_size[1], self.gf_dim]
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        deconv1_2 = relu(deconv2d(relu(tf.add(depool1, res_connect[0])),
                                  output_shape = shapel1, k_h = 3, k_w = 3, d_h = 1, d_w = 1,
                                  name = 'dec_deconv1_2', reuse = reuse))
        xtp1 = tanh(deconv2d(deconv1_2, output_shape = shapeout1, k_h = 3, k_w = 3,
                             d_h = 1, d_w = 1, name = 'dec_deconv1_1', reuse = reuse))
        return xtp1

    def discriminator(self, image):
        h0 = lrelu(conv2d(image, self.df_dim, name = 'dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name = 'dis_h1_conv'),
                              "bn1"))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name = 'dis_h2_conv'),
                              "bn2"))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, name = 'dis_h3_conv'),
                              "bn3"))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "MCNET.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step = step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print(" Loaded model: " + str(model_name))
            return True, model_name
        else:
            return False, None

        # Execute the forward and the backward pass

    def run_single_step(self, global_step):
        print("global_step:", global_step)
        try:
            train_batch = self.sess.run(self.train_iterator.get_next())
            # z=np.random.uniform(-1,1,size=(self.batch_size,self.nz))
            x = self.sess.run([self.x], feed_dict = {self.x: train_batch["images"]})
            _, g_sum = self.sess.run([self.g_optim, self.g_sum], feed_dict = {self.x: train_batch["images"]})
            _, d_sum = self.sess.run([self.d_optim, self.d_sum], feed_dict = {self.x: train_batch["images"]})

            gen_data, train_loss = self.sess.run([self.gen_data, self.total_loss],
                                                       feed_dict = {self.x: train_batch["images"]})

        except tf.errors.OutOfRangeError:
            print("train out of range error")

        try:
            val_batch = self.sess.run(self.val_iterator.get_next())
            val_loss = self.sess.run([self.total_loss], feed_dict = {self.x: val_batch["images"]})
            # self.val_writer.add_summary(val_summary, global_step)
        except tf.errors.OutOfRangeError:
            print("train out of range error")

        return train_loss, val_total_loss



