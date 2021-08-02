__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong,Yanji"
__date__ = "2021-04-13"

from model_modules.video_prediction.models.model_helpers import set_and_check_pred_frames
import tensorflow as tf
from model_modules.video_prediction.layers import layer_def as ld
from model_modules.video_prediction.layers.BasicConvLSTMCell import BasicConvLSTMCell
from tensorflow.contrib.training import HParams
from .vanilla_convLSTM_model import VanillaConvLstmVideoPredictionModel

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
        self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
        self.max_epochs = self.hparams.max_epochs
        self.loss_fun = self.hparams.loss_fun
        self.batch_size = self.hparams.batch_size
        self.recon_weight = self.hparams.recon_weight
        self.bd1 = batch_norm(name = "dis1")
        self.bd2 = batch_norm(name = "dis2")
        self.bd3 = batch_norm(name = "dis3")   

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
        self.inputs = x
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
    
    def get_noise(self):
        """
        Function for creating noise: Given the dimensions (n_batch,n_seq, n_height, n_width, channel)
        """ 
        self.noise = tf.random.uniform(minval=-1., maxval=1., shape=[self.batch_size, self.sequence_length, self.height, self.width, self.channels])
        return self.noise
     
    @staticmethod
    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak*x)

    @staticmethod    
    def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias
     
    @staticmethod
    def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

    @staticmethod
    def bn(x, scope):
        return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        scope=scope)

    def generator(self):
        """
        Function to build up the generator architecture
        args:
            input images: a input tensor with dimension (n_batch,sequence_length,height,width,channel)
        """
        with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
            layer_gen = self.convLSTM_network(self.x)
            layer_gen_pred = layer_gen[:,self.context_frames-1:,:,:,:]
        return layer_gen


    def discriminator(self,vid):
        """
        Function that get discriminator architecture      
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv3d(vid,64,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",name="dis1")
            conv1 = ConvLstmGANVideoPredictionModel.lrelu(conv1)
            conv2 = tf.layers.conv3d(conv1,128,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",name="dis2")
            conv2 = ConvLstmGANVideoPredictionModel.lrelu(self.bd1(conv2))
            conv3 = tf.layers.conv3d(conv2,256,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",name="dis3")
            conv3 = ConvLstmGANVideoPredictionModel.lrelu(self.bd2(conv3))
            conv4 = tf.layers.conv3d(conv3,512,kernel_size=[4,4,4],strides=[2,2,2],padding="SAME",name="dis4")
            conv4 = ConvLstmGANVideoPredictionModel.lrelu(self.bd3(conv4))
            conv5 = tf.layers.conv3d(conv4,1,kernel_size=[2,4,4],strides=[1,1,1],padding="SAME",name="dis5")
            conv5 = tf.reshape(conv5, [-1,1])
            conv5sigmoid = tf.nn.sigmoid(conv5)
            return conv5sigmoid,conv5

    def discriminator0(self,image):
        """
        Function that get discriminator architecture      
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            layer_disc = self.convLSTM_network(image)
            layer_disc = layer_disc[:,self.context_frames-1:self.context_frames,:,:, 0:1]
        return layer_disc

    def discriminator1(self,sequence):
        """
        https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
        Function that give the possibility of a sequence of frames is ture of false 
        the input squence shape is like [batch_size,time_seq_length,height,width,channel]  (e.g., self.x[:,:self.context_frames,:,:,:])
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            print(sequence.shape)
            x = sequence[:,:,:,:,0:1] # extract targeted variable
            x = tf.transpose(x, [0,2,3,1,4]) # sequence shape is like: [batch_size,height,width,time_seq_length]
            x = tf.reshape(x,[x.shape[0],x.shape[1],x.shape[2],x.shape[3]])
            print(x.shape)
            net = ConvLstmGANVideoPredictionModel.lrelu(ConvLstmGANVideoPredictionModel.conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = ConvLstmGANVideoPredictionModel.lrelu(ConvLstmGANVideoPredictionModel.bn(ConvLstmGANVideoPredictionModel.conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'),scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = ConvLstmGANVideoPredictionModel.lrelu(ConvLstmGANVideoPredictionModel.bn(ConvLstmGANVideoPredictionModel.linear(net, 1024, scope='d_fc3'),scope='d_bn3'))
            out_logit = ConvLstmGANVideoPredictionModel.linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)
            print(out.shape)
        return out, out_logit

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
        Return the loss of generator given inputs
        """
        real_labels = tf.ones_like(self.D_fake)
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=real_labels))
        return self.G_loss         
   
    def get_vars(self):
        """
        Get trainable variables from discriminator and generator
        """
        print("trinable_varialbes", len(tf.trainable_variables()))
        self.disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        self.gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        print("self.disc_vars",self.disc_vars)
        print("self.gen_vars",self.gen_vars)
 
  
    def define_gan(self):
        """
        Define gan architectures
        """
        self.noise = self.get_noise()
        self.gen_images = self.generator()
        #!!!! the input of discriminator should be changed when use different discriminators
        self.D_real, self.D_real_logits = self.discriminator(self.x[:,self.context_frames:,:,:,:])
        self.D_fake, self.D_fake_logits = self.discriminator(self.gen_images[:,self.context_frames-1:,:,:,:])
        self.get_gen_loss()
        self.get_disc_loss()
        self.get_vars()
        if self.loss_fun == "rmse":
            self.recon_loss = tf.reduce_mean(tf.square(self.x[:, self.context_frames:,:,:,0] - self.gen_images[:,self.context_frames-1:,:,:,0]))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(self.x[:, self.context_frames:,:,:,0],[-1])
            x_hat_predict_frames_flatten = tf.reshape(self.gen_images[:,self.context_frames-1:,:,:,0],[-1])
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
        print('x_hat shape is: ',x_hat.shape)
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
        self.x_hat= tf.transpose(x_hat, [1, 0, 2, 3, 4])  # change first dim with sec dim  ???? yan: why?
        print('self.x_hat shape is: ',self.x_hat.shape)
        return self.x_hat
     
   
   
