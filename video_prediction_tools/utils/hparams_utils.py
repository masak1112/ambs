#PDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-03-17"



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#auxiliary functions used for parsing the hyerparameters from hparams_dict
def reduce_dict(dict_in: dict, dict_ref: dict):
    """
    Reduces input dictionary to keys from reference dictionary. If the input dictionary lacks some keys, these are 
    copied over from the reference dictionary, i.e. the reference dictionary provides the defaults
    :param dict_in: input dictionary
    :param dict_ref: reference dictionary
    :return: reduced form of input dictionary (with keys complemented from dict_ref if necessary)
    """
    method = reduce_dict.__name__

    # sanity checks
    assert isinstance(dict_in, dict), "%{0}: dict_in must be a dictionary, but is of type {1}"\
                                      .format(method, type(dict_in))
    assert isinstance(dict_ref, dict), "%{0}: dict_ref must be a dictionary, but is of type {1}"\
                                       .format(method, type(dict_ref)) 

    dict_merged = {**dict_ref, **dict_in}
    dict_reduced = {key: dict_merged[key] for key in dict_ref}

    return dict_reduced




