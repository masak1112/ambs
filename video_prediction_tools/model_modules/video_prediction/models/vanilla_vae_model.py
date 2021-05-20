
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-09-01"

from model_helpers import set_and_check_pred_frames
import tensorflow as tf
from model_modules.video_prediction.layers import layer_def as ld
from tensorflow.contrib.training import HParams


class VanillaVAEVideoPredictionModel(object):
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
        self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
        self.max_epochs = self.hparams.max_epochs
        self.nz = self.hparams.nz
        self.loss_fun = self.hparams.loss_fun
        self.batch_size = self.hparams.batch_size 
        self.shuffle_on_val = self.hparams.shuffle_on_val
        self.weight_recon = self.hparams.weight_recon 
        
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
            sequence_length=24,
            max_epochs = 20,
            batch_size = 4,
            lr = 0.001,
            nz = 16,
            loss_fun = "cross_entropy",
            weight_recon = 1,
            shuffle_on_val= True,
        )
        return hparams


    def build_graph(self,x):  
        self.x = x["images"]
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        self.x_hat, self.z_log_sigma_sq, self.z_mu = self.vae_arc_all()
        #This is the loss function (RMSE):
        #This is loss function only for 1 channel (temperature RMSE)
        if self.loss_fun == "rmse":
            self.recon_loss = tf.reduce_mean(tf.square(self.x[:,self.context_frames:,:,:,0] - self.x_hat[:,self.context_frames-1:,:,:,0]))
        elif self.loss_fun == "cross_entropy":
            x_flatten = tf.reshape(self.x[:, self.context_frames:,:,:,0],[-1])
            x_hat_predict_frames_flatten = tf.reshape(self.x_hat[:,self.context_frames-1:,:,:,0],[-1])
            bce = tf.keras.losses.BinaryCrossentropy()
            self.recon_loss = bce(x_flatten,x_hat_predict_frames_flatten)
        else:
            raise ValueError("Loss function is not selected properly, you should chose either 'rmse' or 'cross_entropy'")        
        
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) -tf.exp(self.z_log_sigma_sq), axis=1)
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
        """
        VAE for one timestamp of sequence
        args:
             x      : input tensor, shape is [batch_size,height, width, channel]
             l_name :  int, default is 0, the sequence index
             nz     :  int, default is 16, the latent space 
        return 
             x_hat  :  tensor, is the predicted value 
             z_mu   :  tensor, means values of latent space 
             z_log_sigma_sq: sensor, the variances of latent space
             z      :  tensor, the normal distribution with z_mu, z-log_sigma_sq

        """
        input_shape = x.get_shape().as_list()
        input_width = input_shape[2]
        input_height = input_shape[1]
        print("input_heights:",input_height)
        seq_name = "sq_" + str(l_name) + "_"
        conv1 = ld.conv_layer(inputs=x, kernel_size=3, stride=2, num_features=8, idx=seq_name + "encode_1")
        conv1_shape = conv1.get_shape().as_list()
        print("conv1_shape:",conv1_shape)
        assert conv1_shape[3] == 8 #Features check
        assert conv1_shape[1] == int((input_height - 3 + 1)/2) + 1 #[(Input_volumn - kernel_size + padding)/stride] + 1
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
        x_hat_shape = x_hat.get_shape().as_list()
        pred_height = x_hat_shape[1]
        pred_width = x_hat_shape[2]
        assert pred_height == input_height
        assert pred_width == input_width
        return x_hat, z_mu, z_log_sigma_sq, z

    def vae_arc_all(self):
        """
        Build architecture for all the sequences
        """
        X = []
        z_log_sigma_sq_all = []
        z_mu_all = []
        for i in range(self.sequence_length-1):
            q, z_mu, z_log_sigma_sq, z = VanillaVAEVideoPredictionModel.vae_arc3(self.x[:, i, :, :, :], l_name=i, nz=self.nz)
            X.append(q)
            z_log_sigma_sq_all.append(z_log_sigma_sq)
            z_mu_all.append(z_mu)
        x_hat = tf.stack(X, axis = 1)
        x_hat_shape = x_hat.get_shape().as_list()
        print("x-ha-shape:",x_hat_shape)
        z_log_sigma_sq_all = tf.stack(z_log_sigma_sq_all, axis = 1)
        z_mu_all = tf.stack(z_mu_all, axis = 1)
        return x_hat, z_log_sigma_sq_all, z_mu_all



