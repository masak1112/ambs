# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Some auxilary routines which may are used throughout the project.
Provides:   * get_unique_vars
            * add_str_to_path
            * is_integer
            * ensure_list
            * isw
            * check_str_in_list
            * check_dir
            * reduce_dict
            * provide_default
"""

# import modules
from typing import List, Union
import os
import numpy as np

str_or_List = Union[List, str]
# routines


def get_unique_vars(varnames: List[str]):
    """
    :param varnames: list of variable names (or any other list of strings)
    :return: list with unique elements of inputted varnames list
    """
    vars_uni, varsind = np.unique(varnames, return_index=True)
    nvars_uni = len(vars_uni)

    return vars_uni, varsind, nvars_uni


def add_str_to_path(path_in: str, add_str: str):
    """
    :param path_in: input path which is a candidate to be extended by add_str (see below)
    :param add_str: String to be added to path_in if not already done
    :return: Extended version of path_in (by add_str) if add_str is not already part of path_in.
             Function is also capable to handle carriage returns for handling input-strings obtained by reading a file.
    """

    l_linebreak = path_in.endswith("\n")  # flag for carriage return at the end of input string
    line_str = path_in.rstrip("\n")

    if (not line_str.endswith(add_str)) or \
            (not line_str.endswith(add_str.rstrip("/"))):

        line_str = "{0}{1}/".format(line_str, add_str)
    else:
        print("{0} is already part of {1}. No change is performed.".format(add_str, line_str))

    if l_linebreak:  # re-add carriage return to string if required
        return "{0} \n".format(line_str)
    else:
        return line_str


def is_integer(n):
    """
    :param n: input string
    :return: True if n is a string containing an integer, else False
    """

    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def ensure_list(x):
    """
    Converts input generically to list-object
    :param x: the input data (can be a list, a number/string or an array)
    """
    method = ensure_list.__name__

    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    
    try:
        return list(x)
    except TypeError:
        try:
            return [x]
        except:
            raise TypeError("%{0}: Failed to put input into list.".format(method))


def isw(value, interval):
    """
    Checks if value lies within given interval
    :param value: The value to be checked
    :param interval: The interval defined by lower and upper bound
    :return status: True if value lies in interval
    """

    method = isw.__name__

    if np.shape(interval)[0] != 2:
        raise ValueError("%{0}: interval must contain two values.".format(method))

    if interval[1] <= interval[0]:
        raise ValueError("%{0}: Second value of interval must be larger than first value.".format(method))

    try:
        if interval[0] <= value <= interval[1]:
            status = True
        else:
            status = False
    except Exception as err:
        raise ValueError("%{0}: Could not handle passed value".format(method))

    return status


def check_str_in_list(list_in: List, str2check: str_or_List, labort: bool = True, return_ind: bool = False):
    """
    Checks if all strings are found in list
    :param list_in: input list
    :param str2check: string or list of strings to be checked if they are part of list_in
    :param labort: Flag if error will be risen in case of missing string in list
    :param return_ind: Flag if index for each string found in list will be returned
    :return: True if existence of all strings was confirmed, if return_ind is True, the index of each string in list is
             returned as well
    """
    method = check_str_in_list.__name__

    stat = False
    if isinstance(str2check, str):
        str2check = [str2check]
    elif isinstance(str2check, list):
        assert np.all([isinstance(str1, str) for str1 in str2check]) == True, \
            "Not all elements of str2check are strings"
    else:
        raise ValueError("%{0}: str2check argument must be either a string or a list of strings".format(method))

    stat_element = [True if str1 in list_in else False for str1 in str2check]

    if np.all(stat_element):
        stat = True
    else:
        print("%{0}: The following elements are not part of the input list:".format(method))
        inds_miss = np.where(list(~np.array(stat_element)))[0]
        for i in inds_miss:
            print("* index {0:d}: {1}".format(i, str2check[i]))
        if labort:
            raise ValueError("%{0}: Could not find all expected strings in list.".format(method))
    # return
    if stat and not return_ind:
        return stat
    elif stat:
        return stat, [list_in.index(str_curr) for str_curr in str2check]
    else:
        return stat, []


def check_dir(path2dir: str, lcreate: bool = False):
    """
    Checks if path2dir exists and create it if desired
    :param path2dir:
    :param lcreate: create directory if it is not existing
    :return: True in case of success
    """
    method = check_dir.__name__

    if (path2dir is None) or not isinstance(path2dir, str):
        raise ValueError("%{0}: path2dir must be a string defining a pat to a directory.".format(method))

    elif os.path.isdir(path2dir):
        return True
    else:
        if lcreate:
            try:
                os.makedirs(path2dir)
            except Exception as err:
                print("%{0}: Failed to create directory '{1}'".format(method, path2dir))
                raise err
            print("%{0}: Created directory '{1}'".format(method, path2dir))
            return True
        else:
            raise NotADirectoryError("%{0}: Directory '{1}' does not exist".format(method, path2dir))


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


def find_key(dict_in: dict, key: str):
    """
    Searchs through nested dictionaries for key.
    :param dict_in: input dictionary (cas also be an OrderedDictionary)
    :param key: key to be retrieved
    :return: value of the key in dict_in
    """
    method = find_key.__name__
    # sanity check
    if not isinstance(dict_in, dict):
        raise TypeError("%{0}: dict_in must be a dictionary instance, but is of type '{1}'"
                        .format(method, type(dict_in)))
    # check for key
    if key in dict_in:
        return dict_in[key]
    for k, v in dict_in.items():
        if isinstance(v,dict):
            item = find_key(v, key)
            if item is not None:
                return item

    raise ValueError("%{0}: {1} could not be found in dict_in".format(method, key))


def provide_default(dict_in: dict, keyname: str, default=None, required: bool = False):
    """
    Returns values of key from input dictionary or alternatively its default

    :param dict_in: input dictionary
    :param keyname: name of key which should be added to dict_in if it is not already existing
    :param default: default value of key (returned if keyname is not present in dict_in)
    :param required: Forces existence of keyname in dict_in (otherwise, an error is returned)
    :return: value of requested key or its default retrieved from dict_in
    """
    method = provide_default.__name__

    if not required and default is None:
        raise ValueError("%{0}: Provide default when existence of key in dictionary is not required.".format(method))

    if keyname not in dict_in.keys():
        if required:
            print(dict_in)
            raise ValueError("%{0}: Could not find '{1}' in input dictionary.".format(method, keyname))
        return default
    else:
        return dict_in[keyname]


def depth2intensity(depth, interval=600):
    """
    Function for convertion rainfall depth (in mm) to
    rainfall intensity (mm/h)

    Args:
        depth: float
        float or array of float
        rainfall depth (mm)

        interval : number
        time interval (in sec) which is correspondend to depth values

    Returns:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)
    """
    return depth * 3600 / interval


def intensity2depth(intensity, interval=600):
    """
    Function for convertion rainfall intensity (mm/h) to
    rainfall depth (in mm)

    Args:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)

        interval : number
        time interval (in sec) which is correspondend to depth values

    Returns:
        depth: float
        float or array of float
        rainfall depth (mm)
    """
    return intensity * interval / 3600


def RYScaler(X_mm):
    '''
    Scale RY data from mm (in float64) to brightness (in uint8).

    Args:
        X (numpy.ndarray): RY radar image

    Returns:
        numpy.ndarray(uint8): brightness integer values from 0 to 255
                              for corresponding input rainfall intensity
        float: c1, scaling coefficient
        float: c2, scaling coefficient

    '''
    def mmh2rfl(r, a=256., b=1.42):
        '''
        .. based on wradlib.zr.r2z function

        .. r --> z
        '''
        return a * r ** b

    def rfl2dbz(z):
        '''
        .. based on wradlib.trafo.decibel function

        .. z --> d
        '''
        return 10. * np.log10(z)

    # mm to mm/h
    X_mmh = depth2intensity(X_mm)
    # mm/h to reflectivity
    X_rfl = mmh2rfl(X_mmh)
    # remove zero reflectivity
    # then log10(0.1) = -1 not inf (numpy warning arised)
    X_rfl[X_rfl == 0] = 0.1
    # reflectivity to dBz
    X_dbz = rfl2dbz(X_rfl)
    # remove all -inf
    X_dbz[X_dbz < 0] = 0

    # MinMaxScaling
    c1 = X_dbz.min()
    c2 = X_dbz.max()

    return ((X_dbz - c1) / (c2 - c1) * 255).astype(np.uint8), c1, c2


def inv_RYScaler(X_scl, c1, c2):
    '''
    Transfer brightness (in uint8) to RY data (in mm).
    Function which is inverse to Scaler() function.

    Args:
        X_scl (numpy.ndarray): array of brightness integers obtained
                               from Scaler() function.
        c1: first scaling coefficient obtained from Scaler() function.
        c2: second scaling coefficient obtained from Scaler() function.

    Returns:
        numpy.ndarray(float): RY radar image

    '''
    def dbz2rfl(d):
        '''
        .. based on wradlib.trafo.idecibel function

        .. d --> z
        '''
        return 10. ** (d / 10.)

    def rfl2mmh(z, a=256., b=1.42):
        '''
        .. based on wradlib.zr.z2r function

        .. z --> r
        '''
        return (z / a) ** (1. / b)

    # decibels to reflectivity
    X_rfl = dbz2rfl((X_scl / 255)*(c2 - c1) + c1)
    # 0 dBz are 0 reflectivity, not 1
    X_rfl[X_rfl == 1] = 0
    # reflectivity to rainfall in mm/h
    X_mmh = rfl2mmh(X_rfl)
    # intensity in mm/h to depth in mm
    X_mm = intensity2depth(X_mmh)

    return X_mm

