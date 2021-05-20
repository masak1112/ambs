__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"
__date__ = "2020-11-05"

from model_helpers import set_and_check_pred_frames
import tensorflow as tf
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
        self.predict_frames = set_and_check_pred_frames(self.sequence_length, self.context_frames)
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
            max_epochs=20,
            batch_size=40,
            lr=0.001,
            loss_fun="cross_entropy",
            shuffle_on_val=True,
        )
        return hparams


    def build_graph(self, x):
        self.is_build_graph = False
        self.inputs = x
        self.x = x["images"]
        self.global_step = tf.train.get_or_create_global_step()
        original_global_variables = tf.global_variables()
        # ARCHITECTURE
        self.convLSTM_network()
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

    def convLSTM_network(self):
        network_template = tf.make_template('network',
                                            VanillaConvLstmVideoPredictionModel.convLSTM_cell)  # make the template to share the variables
        # create network
        x_hat = []
        
        #This is for training (optimization of convLSTM layer)
        hidden_g = None
        for i in range(self.sequence_length-1):
            if i < self.context_frames:
                x_1_g, hidden_g = network_template(self.x[:, i, :, :, :], hidden_g)
            else:
                x_1_g, hidden_g = network_template(x_1_g, hidden_g)
            x_hat.append(x_1_g)

        # pack them all together
        x_hat = tf.stack(x_hat)
        self.x_hat= tf.transpose(x_hat, [1, 0, 2, 3, 4])  # change first dim with sec dim
        self.x_hat_predict_frames = self.x_hat[:,self.context_frames-1:,:,:,:]

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

    @staticmethod
    def set_and_check_pred_frames(seq_length, context_frames):
        """
        Checks if sequence length and context_frames are set properly and returns number of frames to be predicted.
        :param seq_length: number of frames/images per sequences
        :param context_frames: number of context frames/images
        :return: number of predicted frames
        """

        method = VanillaConvLstmVideoPredictionModel.set_and_check_pred_frames.__name__

        # sanity checks
        assert isinstance(seq_length, int), "%{0}: Sequence length (seq_length) must be an integer".format(method)
        assert isinstance(context_frames, int), "%{0}: Number of context frames must be an integer".format(method)

        if seq_length > context_frames:
            return seq_length-context_frames
        else:
            raise ValueError("%{0}: Sequence length ({1}) must be larger than context frames ({2})."
                             .format(method, seq_length, context_frames))
