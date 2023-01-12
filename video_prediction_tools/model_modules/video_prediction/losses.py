# SPDX-FileCopyrightText: 2018, alexlee-gk
#
# SPDX-License-Identifier: MIT

import tensorflow as tf


def l1_loss(pred, target):
    return tf.reduce_mean(tf.abs(target - pred))


def l2_loss(pred, target):
    return tf.reduce_mean(tf.square(target - pred))


def normalize_tensor(tensor, eps=1e-10):
    norm_factor = tf.norm(tensor, axis=-1, keepdims=True)
    return tensor / (norm_factor + eps)


def cosine_distance(tensor0, tensor1, keep_axis=None):
    tensor0 = normalize_tensor(tensor0)
    tensor1 = normalize_tensor(tensor1)
    return tf.reduce_mean(tf.reduce_sum(tf.square(tensor0 - tensor1), axis=-1)) / 2.0
                                

def charbonnier_loss(x, epsilon=0.001):
    return tf.reduce_mean(tf.sqrt(tf.square(x) + tf.square(epsilon)))


def gan_loss(logits, labels, gan_loss_type):
    # use 1.0 (or 1.0 - discrim_label_smooth) for real data and 0.0 for fake data
    if gan_loss_type == 'GAN':
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if labels in (0.0, 1.0):
            labels = tf.constant(labels, dtype=logits.dtype, shape=logits.get_shape())
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            loss = tf.reduce_mean(sigmoid_kl_with_logits(logits, labels))
    elif gan_loss_type == 'LSGAN':
        # discrim_loss = tf.reduce_mean((tf.square(predict_real - 1) + tf.square(predict_fake)))
        # gen_loss = tf.reduce_mean(tf.square(predict_fake - 1))
        loss = tf.reduce_mean(tf.square(logits - labels))
    elif gan_loss_type == 'SNGAN':
        # this is the form of the loss used in the official implementation of the SNGAN paper, but it leads to
        # worse results in our video prediction experiments
        if labels == 0.0:
            loss = tf.reduce_mean(tf.nn.softplus(logits))
        elif labels == 1.0:
            loss = tf.reduce_mean(tf.nn.softplus(-logits))
        else:
            raise NotImplementedError
    else:
        raise ValueError('Unknown GAN loss type %s' % gan_loss_type)
    return loss

