# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong,Yanji"
__date__ = "2021-04-13"

from model_modules.video_prediction.models.model_helpers import set_and_check_pred_frames
import tensorflow as tf
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.layers.BasicConvLSTMCell import BasicConvLSTMCell
from tensorflow.contrib.training import HParams
from general_utils import reduce_dict

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
        self.ngf = self.hparams.ngf
        self.ndf = self.hparams.ndf
        self.total_loss = None
        self.context_frames = self.hparams.context_frames
        self.sequence_length = self.hparams.sequence_length
        self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
        self.max_epochs = self.hparams.max_epochs
        self.loss_fun = self.hparams.loss_fun
        self.batch_size = self.hparams.batch_size
        self.recon_weight = self.hparams.recon_weight
        self.bd1 = batch_norm(name = "bd1")
        self.bd2 = batch_norm(name = "bd2")

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self):
        """
        Parse the hparams setting to ovoerride the default ones
        """
        self.hparams_dict = reduce_dict(self.hparams_dict, self.get_default_hparams().values())
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
            max_epochs = 2,
            batch_size = 4,
            lr = 0.001,
            loss_fun = "rmse",
            shuffle_on_val= True,
            recon_weight=0.99,
            ngf = 4,
            ndf = 4
         )
        return hparams

    def build_graph(self, x: tf.Tensor) ->bool:
        self.is_build_graph = False
        self.inputs = x
        print('self.inputs: {}'.format(self.inputs))
        self.width = 92#self.inputs.shape.as_list()[3]
        self.height = 46#self.inputs.shape.as_list()[2]
        self.channels = 1#self.inputs.shape.as_list()[4]
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        # Architecture
        self.define_gan()
        self.total_loss = (1-self.recon_weight) * self.G_loss + self.recon_weight*self.recon_loss
        self.D_loss =  (1-self.recon_weight) * self.D_loss
        if self.mode == "train":
            if self.recon_weight == 1:
                print("Only train generator- convLSTM") 
                self.train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.total_loss, var_list=self.gen_vars) 
            else:
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

    @staticmethod
    def Unet_ConvLSTM_cell(x: tf.Tensor, ngf: int, hidden: tf.Tensor) -> tf.Tensor:
        """
        Build up a Unet ConvLSTM cell for each time stamp i

        params: x:     the input at timestamp i
        params: ngf:   the numnber of filters for convoluational layers
        params: hidden: the hidden state from the previous timestamp t-1
        params: cell_id: the cell layer id
        return:
               outputs: the predict frame at timestamp i
               hidden:  the hidden state at current timestamp i
        """
        input_shape = x.get_shape().as_list()
        num_channels = input_shape[3]
        with tf.variable_scope("down_scale", reuse = tf.AUTO_REUSE):
            conv1f = ld.conv_layer(x, 3 , 1, ngf, 1, initializer=tf.contrib.layers.xavier_initializer(), activate="relu")
            conv1s = ld.conv_layer(conv1f, 3, 1, ngf, 2, initializer=tf.contrib.layers.xavier_initializer(), activate="relu")
            pool1 = tf.layers.max_pooling2d(conv1s, pool_size=(2, 2), strides=(2, 2))
            print('pool1 shape: ',pool1.shape)

            conv2f = ld.conv_layer(pool1, 3, 1, ngf * 2, 3, initializer=tf.contrib.layers.xavier_initializer(), activate="relu")
            conv2s = ld.conv_layer(conv2f, 3, 1, ngf * 2, 4, initializer = tf.contrib.layers.xavier_initializer(), activate = "relu")
            pool2 = tf.layers.max_pooling2d(conv2s, pool_size=(2, 2), strides=(2, 2))
            print('pool2 shape: ',pool2.shape)

            conv3f = ld.conv_layer(pool2, 3, 1, ngf * 4, 5, initializer=tf.contrib.layers.xavier_initializer(), activate="relu")
            conv3s = ld.conv_layer(conv3f, 3, 1, ngf * 4, 6, initializer = tf.contrib.layers.xavier_initializer(), activate = "relu")
            pool3 = tf.layers.max_pooling2d(conv3s, pool_size=(2, 2), strides=(2, 2))
            print('pool3 shape: ',pool3.shape)

            convLSTM_input = pool3
            #convLSTM_input = tf.layers.dropout(pool2, 0.8)

        convLSTM4, hidden = ConvLstmGANVideoPredictionModel.convLSTM_cell(convLSTM_input, hidden)
        print('convLSTM4 shape: ',convLSTM4.shape)
  
        with tf.variable_scope("upscale", reuse = tf.AUTO_REUSE):
            deconv5 = ld.transpose_conv_layer(convLSTM4, 2, 2, ngf * 4, 1, initializer=tf.contrib.layers.xavier_initializer(), activate="relu")
            print('deconv5 shape: ',deconv5.shape)
            up5 = tf.concat([deconv5, conv3s], axis=3)
            print('up5 shape: ',up5.shape)

            conv5f = ld.conv_layer(up5, 3, 1, ngf * 4, 2, initializer = tf.contrib.layers.xavier_initializer(), activate="relu")
            conv5s = ld.conv_layer(conv5f, 3, 1, ngf * 4, 3, initializer = tf.contrib.layers.xavier_initializer(), activate="relu")
            print('conv5s shape:',conv5s.shape)

            deconv6 = ld.transpose_conv_layer(conv5s, 2, 2, ngf * 2, 4, initializer=tf.contrib.layers.xavier_initializer(), activate="relu")
            print('deconv6 shape: ',deconv6.shape)
            up6 = tf.concat([deconv6, conv2s], axis=3)
            print('up6 shape: ',up6.shape)

            conv6f = ld.conv_layer(up6, 3, 1, ngf * 2, 5, initializer = tf.contrib.layers.xavier_initializer(), activate="relu")
            conv6s = ld.conv_layer(conv6f, 3, 1, ngf * 2, 6, initializer = tf.contrib.layers.xavier_initializer(), activate="relu")
            print('conv6s shape:',conv6s.shape)

            deconv7 = ld.transpose_conv_layer(conv6s, 2, 2, ngf, 7, initializer = tf.contrib.layers.xavier_initializer(), activate="relu")
            print('deconv7 shape: ',deconv7.shape)
            up7 = tf.concat([deconv7, conv1s], axis=3)
            print('up7 shape: ',up7.shape)

            conv7f = ld.conv_layer(up7, 3, 1, ngf, 8, initializer = tf.contrib.layers.xavier_initializer(), activate="relu")
            conv7s = ld.conv_layer(conv7f, 3, 1, ngf, 9, initializer = tf.contrib.layers.xavier_initializer(),activate= "relu")
            print('conv7s shape:',conv7s.shape)

            conv7t = ld.conv_layer(conv7s, 3, 1, num_channels, 10, initializer = tf.contrib.layers.xavier_initializer(),activate="relu")
            outputs = ld.conv_layer(conv7t, 1, 1, num_channels, 11, initializer = tf.contrib.layers.xavier_initializer(),activate="linear")
            print('outputs shape: ',outputs.shape)
        return outputs, hidden

    def generator(self, x: tf.Tensor) -> tf.Tensor:
        """
        Function to build up the generator architecture, here we take Unet_ConvLSTM as generator
        args:
            input images: a input tensor with dimension (n_batch,sequence_length,height,width,channel)
            output images: (n_batch,forecast_length,height,width,channel)
        """
        network_template = tf.make_template('network', ConvLstmGANVideoPredictionModel.Unet_ConvLSTM_cell)
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
            # create network
            x_hat = []
            #This is for training (optimization of convLSTM layer)
            hidden_g = None
            for i in range(self.sequence_length-1):
                print('i: ',i)
                if i < self.context_frames:
                    x_1_g, hidden_g = network_template(x[:, i, :, :, :], self.ngf, hidden_g)
                else:
                    x_1_g, hidden_g = network_template(x_1_g, self.ngf, hidden_g)
                x_hat.append(x_1_g)
            # pack them all together
            x_hat = tf.stack(x_hat)
            self.x_hat= tf.transpose(x_hat, [1, 0, 2, 3, 4])
            print('self.x_hat shape is: ',self.x_hat.shape)
        return self.x_hat

    def discriminator(self,x):
        """
        Function that get discriminator architecture      
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv3d(x, 4, kernel_size=[4,4,4], strides=[1,2,2], padding="SAME", name="dis1")
            conv1 = ConvLstmGANVideoPredictionModel.lrelu(conv1)
            #conv2 = tf.layers.conv3d(conv1, 1, kernel_size=[4,4,4], strides=[1,2,2], padding="SAME", name="dis2")
            conv2 = tf.reshape(conv1, [-1,1])
            #fc1 = ConvLstmGANVideoPredictionModel.lrelu(self.bd1(ConvLstmGANVideoPredictionModel.linear(conv2, output_size=256, scope='d_fc1')))
            fc2 = ConvLstmGANVideoPredictionModel.lrelu(self.bd2(ConvLstmGANVideoPredictionModel.linear(conv2, output_size=64, scope='d_fc2')))
            out_logit = ConvLstmGANVideoPredictionModel.linear(fc2, 1, scope='d_fc3')
            out = tf.nn.sigmoid(out_logit)
            #out,out_logit = self.Conv3Dnet(x,self.ndf)
            return out,out_logit

    def get_disc_loss(self):
        """
        Return the loss of discriminator given inputs
        """
        real_labels = tf.ones_like(self.D_real)
        gen_labels = tf.zeros_like(self.D_fake)
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=real_labels))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=gen_labels))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        return self.D_loss

    def get_gen_loss(self):
        """
        Param:
	    num_images    : the number of images the generator should produce, which is also the lenght of the real image
            z_dim     : the dimension of the noise vector, a scalar
        Return the loss of generator given inputs
        """
        real_labels = tf.ones_like(self.D_fake)
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=real_labels))
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
        self.gen_images = self.generator(self.inputs)
        self.D_real, self.D_real_logits = self.discriminator(self.inputs[:,self.context_frames:, :, :, 0:1]) # use the first varibale as targeted
        #self.D_fake, self.D_fake_logits = self.discriminator(self.gen_images[:,:,:,:,0:1]) #0:1
        self.D_fake, self.D_fake_logits = self.discriminator(self.gen_images[:,self.context_frames-1:, :, :, 0:1]) #0:1

        self.get_gen_loss()
        self.get_disc_loss()
        self.get_vars()
        if self.loss_fun == "rmse":
            #self.recon_loss = tf.reduce_mean(tf.square(self.inputs[:, self.context_frames:,:,:,0] - self.gen_images[:,:,:,:,0]))
            self.recon_loss = tf.reduce_mean(tf.square(self.inputs[:, self.context_frames:, :, :, 0] - self.gen_images[:, self.context_frames-1:, :, :, 0]))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(self.inputs[:, self.context_frames:,:,:,0],[-1])
            #x_hat_predict_frames_flatten = tf.reshape(self.gen_images[:,:,:,:,0],[-1])
            x_hat_predict_frames_flatten = tf.reshape(self.gen_images[:,self.context_frames-1:, :, :, 0], [-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            self.recon_loss = bce(x_flatten, x_hat_predict_frames_flatten)
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'rmse' or 'cross_entropy'")

    @staticmethod
    def convLSTM_cell(inputs, hidden):
        y_0 = inputs #we only usd patch 1, but the original paper use patch 4 for the moving mnist case, but use 2 for Radar Echo Dataset
        # conv lstm cell
        cell_shape = y_0.get_shape().as_list()
        channels = cell_shape[-1]
        with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell(shape = [cell_shape[1], cell_shape[2]], filter_size=5, num_features=64)
            if hidden is None:
                hidden = cell.zero_state(y_0, tf.float32)
            output, hidden = cell(y_0, hidden)
        return output, hidden
        #output_shape = output.get_shape().as_list()
        #z3 = tf.reshape(output, [-1, output_shape[1], output_shape[2], output_shape[3]])
        ###we feed the learn representation into a 1 × 1 convolutional layer to generate the final prediction
        #x_hat = ld.conv_layer(z3, 1, 1, channels, "decode_1", activate="sigmoid")
        #print('x_hat shape is: ',x_hat.shape)
        #return x_hat, hidden

    def get_noise(self, x, sigma=0.2):
        """
        Function for creating noise: Given the dimensions (n_batch,n_seq, n_height, n_width, channel)
        """
        x_shape = x.get_shape().as_list()
        noise = sigma * tf.random.uniform(minval=-1., maxval=1., shape=x_shape)
        x = x + noise
        return x

    def Conv3Dnet_v1(self, x, ndf):
        conv1 = tf.layers.conv3d(x, ndf, kernel_size = [4, 4, 4], strides = [1, 2, 2], padding = "SAME", name = 'conv1')
        conv1 = self.lrelu(conv1)
        # conv2 = tf.layers.conv3d(conv1,ndf*2,kernel_size=[4,4,4],strides=[1,2,2],padding="SAME",name='conv2')
        # conv2 = self.lrelu(conv2)
        conv3 = tf.layers.conv3d(conv1, 1, kernel_size = [4, 4, 4], strides = [1, 1, 1], padding = "SAME", name = 'conv3')
        fl = tf.reshape(conv3, [-1, 1])
        print('fl shape: ', fl.shape)
        fc1 = self.lrelu(self.bd1(self.linear(fl, 256, scope = 'fc1')))
        print('fc1 shape: ', fc1.shape)
        fc2 = self.lrelu(self.bd2(self.linear(fc1, 64, scope = 'fc2')))
        print('fc2 shape: ', fc2.shape)
        out_logit = self.linear(fc2, 1, scope = 'out')
        out = tf.nn.sigmoid(out_logit)
        return out, out_logit


    def Conv3Dnet_v2(self, x, ndf):
        """
            args:
            input images: a input tensor with dimension (n_batch,forecast_length,height,width,channel)
            output images:
        """
        conv1 = Conv3D(ndf, 4, strides = (1, 2, 2), padding = 'same', kernel_initializer = 'he_normal')(x)
        bn1 = BatchNormalization()(conv1)
        bn1 = LeakyReLU(0.2)(bn1)
        pool1 = MaxPooling3D(pool_size = (1, 2, 2), padding = 'same')(bn1)
        noise1 = self.get_noise(pool1)

        conv2 = Conv3D(ndf * 2, 4, strides = (1, 2, 2), padding = 'same', kernel_initializer = 'he_normal')(noise1)
        bn2 = BatchNormalization()(conv2)
        bn2 = LeakyReLU(0.2)(bn2)
        pool2 = MaxPooling3D(pool_size = (1, 2, 2), padding = 'same')(bn2)
        noise2 = self.get_noise(pool2)

        conv3 = Conv3D(ndf * 4, 4, strides = (1, 2, 2), padding = 'same', kernel_initializer = 'he_normal')(noise2)
        bn3 = BatchNormalization()(conv3)
        bn3 = LeakyReLU(0.2)(bn3)
        pool3 = MaxPooling3D(pool_size = (1, 2, 2), padding = 'same')(bn3)

        conv4 = Conv3D(1, 4, 1, padding = 'same')(pool3)

        fl = tf.reshape(conv4, [-1, 1])
        drop1 = Dropout(0.3)(fl)
        fc1 = Dense(1024, activation = 'relu')(drop1)
        drop2 = Dropout(0.3)(fc1)
        fc2 = Dense(512, activation = 'relu')(drop2)
        out_logit = Dense(1, activation = 'linear')(fc2)
        out = tf.nn.sigmoid(out_logit)
        return out, out_logit


    @staticmethod
    def lrelu(x, leak=0.2, name='lrelu'):
        return tf.maximum(x, leak * x)


    @staticmethod
    def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
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

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

