# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

import tensorflow as tf
from .layer_def import *

class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self,input, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape
        num_features = self.num_features
        #x= tf.placeholder(tf.float32, shape=[input.shape[0], shape[0], shape[1], num_features * 2])#Bing: add this to
        zeros = tf.zeros([tf.shape(input)[0], shape[0], shape[1], num_features * 2])
        #zeros = tf.zeros_like(x)
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
                 state_is_tuple=False, activation=tf.nn.tanh):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None,reuse=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__,reuse=reuse):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis = 3, num_or_size_splits = 2, value = state)

            input_h = [inputs,h]
            #Bing20200930#replace with non-linear convolutional layers
            #concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)
            input_h_con = tf.concat(axis = 3, values = input_h)
            concat = conv_layer(input_h_con, self.filter_size, 1, self.num_features*4, "decode_1", activate="sigmoid")
            i, j, f, o = tf.split(axis = 3, num_or_size_splits = 4, value = concat)
            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis = 3, values = [new_c, new_h])
            print("new h", new_h)
            print("new state",new_state)
            return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype = dtype)
        if len(args) == 1:
            print("args[0]:",args[0])
            res = tf.nn.conv2d(args[0], matrix, strides = [1, 1, 1, 1], padding = 'SAME')
            print("res1:",res)
        else:
            print("matrix:",matrix)
            print("tf.concat(axis = 3, values = args):",tf.concat(axis = 3, values = args))
            res = tf.nn.conv2d(tf.concat(axis = 3, values = args), matrix, strides = [1, 1, 1, 1], padding = 'SAME')
            print("res2:",res)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype = dtype,
            initializer = tf.constant_initializer(
                bias_start, dtype = dtype))
    return res + bias_term

