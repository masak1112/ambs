# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA, DNA, and STP."""

import itertools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from model_modules.video_prediction.models import VideoPredictionModel
from .sna_model import basic_conv_lstm_cell


# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


def construct_model(images,
                    actions=None,
                    states=None,
                    iter_num=-1.0,
                    kernel_size=(5, 5),
                    k=-1,
                    use_state=True,
                    num_masks=10,
                    stp=False,
                    cdna=True,
                    dna=False,
                    context_frames=2,
                    pix_distributions=None):
    """Build convolutional lstm video predictor using STP, CDNA, or DNA.

    Args:
        images: tensor of ground truth image sequences
        actions: tensor of action sequences
        states: tensor of ground truth state sequences
        iter_num: tensor of the current training iteration (for sched. sampling)
        k: constant used for scheduled sampling. -1 to feed in own prediction.
        use_state: True to include state and action in prediction
        num_masks: the number of different pixel motion predictions (and
                   the number of masks for each of those predictions)
        stp: True to use Spatial Transformer Predictor (STP)
        cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
        dna: True to use Dynamic Neural Advection (DNA)
        context_frames: number of ground truth frames to pass in before
                        feeding in own predictions
    Returns:
        gen_images: predicted future image frames
        gen_states: predicted future states

    Raises:
        ValueError: if more than one network option specified or more than 1 mask
        specified for DNA model.
    """
    DNA_KERN_SIZE = kernel_size[0]

    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')
    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    lstm_func = basic_conv_lstm_cell

    # Generated robot states and images.
    gen_states, gen_images = [], []
    gen_pix_distrib = []
    gen_masks = []
    current_state = states[0]

    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(
            tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
        feedself = False

    # LSTM state sizes and states.
    lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
    lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
    lstm_state5, lstm_state6, lstm_state7 = None, None, None

    for t, action in enumerate(actions):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                # Feed in generated image.
                prev_image = gen_images[-1]
                if pix_distributions is not None:
                    prev_pix_distrib = gen_pix_distrib[-1]
            elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(images[t], gen_images[-1], batch_size,
                                              num_ground_truth)
            else:
                # Always feed in ground_truth
                prev_image = images[t]
                if pix_distributions is not None:
                    prev_pix_distrib = pix_distributions[t]
                    # prev_pix_distrib = tf.expand_dims(prev_pix_distrib, -1)

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])

            enc0 = slim.layers.conv2d(
                prev_image,
                32, [5, 5],
                stride=2,
                scope='scale1_conv1',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm1'})

            hidden1, lstm_state1 = lstm_func(
                enc0, lstm_state1, lstm_size[0], scope='state1')
            hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
            hidden2, lstm_state2 = lstm_func(
                hidden1, lstm_state2, lstm_size[1], scope='state2')
            hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
            enc1 = slim.layers.conv2d(
                hidden2, hidden2.get_shape()[3], [3, 3], stride=2, scope='conv2')

            hidden3, lstm_state3 = lstm_func(
                enc1, lstm_state3, lstm_size[2], scope='state3')
            hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')
            hidden4, lstm_state4 = lstm_func(
                hidden3, lstm_state4, lstm_size[3], scope='state4')
            hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm5')
            enc2 = slim.layers.conv2d(
                hidden4, hidden4.get_shape()[3], [3, 3], stride=2, scope='conv3')

            # Pass in state and action.
            smear = tf.reshape(
                state_action,
                [int(batch_size), 1, 1, int(state_action.get_shape()[1])])
            smear = tf.tile(
                smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
            if use_state:
                enc2 = tf.concat(axis=3, values=[enc2, smear])
            enc3 = slim.layers.conv2d(
                enc2, hidden4.get_shape()[3], [1, 1], stride=1, scope='conv4')

            hidden5, lstm_state5 = lstm_func(
                enc3, lstm_state5, lstm_size[4], scope='state5')  # last 8x8
            hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
            enc4 = slim.layers.conv2d_transpose(
                hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

            hidden6, lstm_state6 = lstm_func(
                enc4, lstm_state6, lstm_size[5], scope='state6')  # 16x16
            hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')
            # Skip connection.
            hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16

            enc5 = slim.layers.conv2d_transpose(
                hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
            hidden7, lstm_state7 = lstm_func(
                enc5, lstm_state7, lstm_size[6], scope='state7')  # 32x32
            hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

            # Skip connection.
            hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

            enc6 = slim.layers.conv2d_transpose(
                hidden7,
                hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm9'})

            if dna:
                # Using largest hidden state for predicting untied conv kernels.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, DNA_KERN_SIZE ** 2, 1, stride=1, scope='convt4')
            else:
                # Using largest hidden state for predicting a new image layer.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, color_channels, 1, stride=1, scope='convt4')
                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                transformed = [tf.nn.sigmoid(enc7)]

            if stp:
                stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
                stp_input1 = slim.layers.fully_connected(
                    stp_input0, 100, scope='fc_stp')

                # disabling capability to generete pixels
                reuse_stp = None
                if reuse:
                    reuse_stp = reuse
                transformed = stp_transformation(prev_image, stp_input1, num_masks, reuse_stp)
                # transformed += stp_transformation(prev_image, stp_input1, num_masks)

                if pix_distributions is not None:
                    transf_distrib = stp_transformation(prev_pix_distrib, stp_input1, num_masks, reuse=True)

            elif cdna:
                cdna_input = tf.reshape(hidden5, [int(batch_size), -1])

                new_transformed, cdna_kerns = cdna_transformation(prev_image,
                                                                  cdna_input,
                                                                  num_masks,
                                                                  int(color_channels),
                                                                  kernel_size,
                                                                  reuse_sc=reuse)
                transformed += new_transformed

                if pix_distributions is not None:
                    if not dna:
                        transf_distrib = [prev_pix_distrib]
                    new_transf_distrib, _ = cdna_transformation(prev_pix_distrib,
                                                                cdna_input,
                                                                num_masks,
                                                                prev_pix_distrib.shape[-1].value,
                                                                kernel_size,
                                                                reuse_sc=True)
                    transf_distrib += new_transf_distrib

            elif dna:
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7, DNA_KERN_SIZE)]

            masks = slim.layers.conv2d_transpose(
                enc6, num_masks + 1, 1, stride=1, scope='convt7')
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
                [int(batch_size), int(img_height), int(img_width), num_masks + 1])
            mask_list = tf.split(masks, num_masks + 1, axis=3)
            output = mask_list[0] * prev_image
            for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask
            gen_images.append(output)
            gen_masks.append(mask_list)

            if dna and pix_distributions is not None:
                transf_distrib = [dna_transformation(prev_pix_distrib, enc7, DNA_KERN_SIZE)]

            if pix_distributions is not None:
                pix_distrib_output = mask_list[0] * prev_pix_distrib
                for layer, mask in zip(transf_distrib, mask_list[1:]):
                    pix_distrib_output += layer * mask
                pix_distrib_output /= tf.reduce_sum(pix_distrib_output, axis=(1, 2), keepdims=True)
                gen_pix_distrib.append(pix_distrib_output)

            if int(current_state.get_shape()[1]) == 0:
                current_state = tf.zeros_like(state_action)
            else:
                current_state = slim.layers.fully_connected(
                    state_action,
                    int(current_state.get_shape()[1]),
                    scope='state_pred',
                    activation_fn=None)
            gen_states.append(current_state)

    return gen_images, gen_states, gen_masks, gen_pix_distrib


## Utility functions
def stp_transformation(prev_image, stp_input, num_masks):
    """Apply spatial transformer predictor (STP) to previous image.

    Args:
        prev_image: previous image to be transformed.
        stp_input: hidden layer to be used for computing STN parameters.
        num_masks: number of masks and hence the number of STP transformations.
    Returns:
        List of images transformed by the predicted STP parameters.
     """
    # Only import spatial transformer if needed.
    from spatial_transformer import transformer

    identity_params = tf.convert_to_tensor(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
    transformed = []
    for i in range(num_masks - 1):
        params = slim.layers.fully_connected(
            stp_input, 6, scope='stp_params' + str(i),
            activation_fn=None) + identity_params
        transformed.append(transformer(prev_image, params))

    return transformed


def cdna_transformation(prev_image, cdna_input, num_masks, color_channels, kernel_size, reuse_sc=None):
    """Apply convolutional dynamic neural advection to previous image.

    Args:
        prev_image: previous image to be transformed.
        cdna_input: hidden lyaer to be used for computing CDNA kernels.
        num_masks: the number of masks and hence the number of CDNA transformations.
        color_channels: the number of color channels in the images.
    Returns:
        List of images transformed by the predicted CDNA kernels.
    """
    batch_size = int(cdna_input.get_shape()[0])
    height = int(prev_image.get_shape()[1])
    width = int(prev_image.get_shape()[2])

    # Predict kernels using linear function of last hidden layer.
    cdna_kerns = slim.layers.fully_connected(
        cdna_input,
        kernel_size[0] * kernel_size[1] * num_masks,
        scope='cdna_params',
        activation_fn=None,
        reuse=reuse_sc)

    # Reshape and normalize.
    cdna_kerns = tf.reshape(
        cdna_kerns, [batch_size, kernel_size[0], kernel_size[1], 1, num_masks])
    cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
    norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keepdims=True)
    cdna_kerns /= norm_factor

    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
    cdna_kerns = tf.reshape(cdna_kerns, [kernel_size[0], kernel_size[1], batch_size, num_masks])
    # Swap the batch and channel dimensions.
    prev_image = tf.transpose(prev_image, [3, 1, 2, 0])

    # Transform image.
    transformed = tf.nn.depthwise_conv2d(prev_image, cdna_kerns, [1, 1, 1, 1], 'SAME')

    # Transpose the dimensions to where they belong.
    transformed = tf.reshape(transformed, [color_channels, height, width, batch_size, num_masks])
    transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
    transformed = tf.unstack(transformed, axis=-1)
    return transformed, cdna_kerns


def dna_transformation(prev_image, dna_input, kernel_size):
    """Apply dynamic neural advection to previous image.

    Args:
        prev_image: previous image to be transformed.
        dna_input: hidden lyaer to be used for computing DNA transformation.
    Returns:
        List of images transformed by the predicted CDNA kernels.
    """
    # Construct translated images.
    pad_along_height = (kernel_size[0] - 1)
    pad_along_width = (kernel_size[1] - 1)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    prev_image_pad = tf.pad(prev_image, [[0, 0],
                                         [pad_top, pad_bottom],
                                         [pad_left, pad_right],
                                         [0, 0]])
    image_height = int(prev_image.get_shape()[1])
    image_width = int(prev_image.get_shape()[2])

    inputs = []
    for xkern in range(kernel_size[0]):
        for ykern in range(kernel_size[1]):
            inputs.append(
                tf.expand_dims(
                    tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                             [-1, image_height, image_width, -1]), [3]))
    inputs = tf.concat(axis=3, values=inputs)

    # Normalize channels to 1.
    kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
    kernel = tf.expand_dims(
        kernel / tf.reduce_sum(
            kernel, [3], keepdims=True), [4])
    return tf.reduce_sum(kernel * inputs, [3], keepdims=False)


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data points.

    Args:
        ground_truth_x: tensor of ground-truth data points.
        generated_x: tensor of generated data points.
        batch_size: batch size
        num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
        New batch with num_ground_truth sampled from ground_truth_x and the rest
        from generated_x.
    """
    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)
    return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])


def generator_fn(inputs, hparams=None):
    images = tf.unstack(inputs['images'], axis=0)
    actions = tf.unstack(inputs['actions'], axis=0)
    states = tf.unstack(inputs['states'], axis=0)
    pix_distributions = tf.unstack(inputs['pix_distribs'], axis=0) if 'pix_distribs' in inputs else None
    iter_num = tf.to_float(tf.train.get_or_create_global_step())

    gen_images, gen_states, gen_masks, gen_pix_distrib = \
        construct_model(images,
                        actions,
                        states,
                        iter_num=iter_num,
                        kernel_size=hparams.kernel_size,
                        k=hparams.schedule_sampling_k,
                        num_masks=hparams.num_masks,
                        cdna=hparams.transformation == 'cdna',
                        dna=hparams.transformation == 'dna',
                        stp=hparams.transformation == 'stp',
                        context_frames=hparams.context_frames,
                        pix_distributions=pix_distributions)
    outputs = {
        'gen_images': tf.stack(gen_images, axis=0),
        'gen_states': tf.stack(gen_states, axis=0),
        'masks': tf.stack([tf.stack(gen_mask_list, axis=-1) for gen_mask_list in gen_masks], axis=0),
    }
    if 'pix_distribs' in inputs:
        outputs['gen_pix_distribs'] = tf.stack(gen_pix_distrib, axis=0)
    gen_images = outputs['gen_images'][hparams.context_frames - 1:]
    return gen_images, outputs


class DNAVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(DNAVideoPredictionModel, self).__init__(
            generator_fn, *args, **kwargs)

    def get_default_hparams_dict(self):
        default_hparams = super(DNAVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            batch_size=32,
            l1_weight=0.0,
            l2_weight=1.0,
            transformation='cdna',
            kernel_size=(9, 9),
            num_masks=10,
            schedule_sampling_k=900.0,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parse_hparams(self, hparams_dict, hparams):
        hparams = super(DNAVideoPredictionModel, self).parse_hparams(hparams_dict, hparams)
        if self.mode == 'test':
            def override_hparams_maybe(name, value):
                orig_value = hparams.values()[name]
                if orig_value != value:
                    print('Overriding hparams from %s=%r to %r for mode=%s.' %
                          (name, orig_value, value, self.mode))
                    hparams.set_hparam(name, value)
            override_hparams_maybe('schedule_sampling_k', -1)
        return hparams
