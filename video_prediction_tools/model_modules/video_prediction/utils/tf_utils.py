import itertools
import os
from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest



def get_checkpoint_restore_saver(checkpoint, var_list=None, skip_global_step=False, restore_to_checkpoint_mapping=None):

    method = get_checkpoint_restore_saver.__name__

    if os.path.isdir(checkpoint):
        # latest_checkpoint doesn't work when the path has special characters
        checkpoint = tf.train.latest_checkpoint(checkpoint)
    # print name of checkpoint-file for verbosity
    print("%{0}: The follwoing checkpoint is used for restoring the model: '{1}'".format(method, checkpoint))
    # Start processing the checkpoint
    checkpoint_reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint)
    checkpoint_var_names = checkpoint_reader.get_variable_to_shape_map().keys()
    restore_to_checkpoint_mapping = restore_to_checkpoint_mapping or (lambda name, _: name.split(':')[0])
    if not var_list:
        var_list = tf.global_variables()
    restore_vars = {restore_to_checkpoint_mapping(var.name, checkpoint_var_names): var for var in var_list}
    if skip_global_step and 'global_step' in restore_vars:
        del restore_vars['global_step']
    # restore variables that are both in the global graph and in the checkpoint
    restore_and_checkpoint_vars = {name: var for name, var in restore_vars.items() if name in checkpoint_var_names}
    #restore_saver = tf.train.Saver(max_to_keep=1, var_list=restore_and_checkpoint_vars, filename=checkpoint)
    # print out information regarding variables that were not restored or used for restoring
    restore_not_in_checkpoint_vars = {name: var for name, var in restore_vars.items() if
                                      name not in checkpoint_var_names}
    checkpoint_not_in_restore_var_names = [name for name in checkpoint_var_names if name not in restore_vars]
    if skip_global_step and 'global_step' in checkpoint_not_in_restore_var_names:
        checkpoint_not_in_restore_var_names.remove('global_step')
    if restore_not_in_checkpoint_vars:
        print("global variables that were not restored because they are "
              "not in the checkpoint:")
        for name, _ in sorted(restore_not_in_checkpoint_vars.items()):
            print("    ", name)
    if checkpoint_not_in_restore_var_names:
        print("checkpoint variables that were not used for restoring "
              "because they are not in the graph:")
        for name in sorted(checkpoint_not_in_restore_var_names):
            print("    ", name)


    restore_saver = tf.train.Saver(max_to_keep=1, var_list=restore_and_checkpoint_vars, filename=checkpoint)

    return restore_saver, checkpoint

