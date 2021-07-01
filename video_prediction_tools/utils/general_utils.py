"""
Some auxilary routines which may are used throughout the project.
Provides:   * get_unique_vars
            * add_str_to_path
            * is_integer
            * isw
            * check_str_in_list
            * check_dir
            * reduce_dict
            * provide_default
            * get_era5_atts
"""

# import modules
import os
import numpy as np
#import xarray as xr

# routines
def get_unique_vars(varnames):
    """
    :param varnames: list of variable names (or any other list of strings)
    :return: list with unique elements of inputted varnames list
    """
    vars_uni, varsind = np.unique(varnames, return_index=True)
    nvars_uni = len(vars_uni)

    return vars_uni, varsind, nvars_uni


def add_str_to_path(path_in, add_str):
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


def check_str_in_list(list_in, str2check, labort=True):
    """
    Checks if all strings are found in list
    :param list_in: input list
    :param str2check: string or list of strings to be checked if they are part of list_in
    :return: True if existence of all strings was confirmed
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

    if not np.all(stat_element):
        print("%{0}: The following elements are not part of the input list:".format(method))
        inds_miss = np.where(stat_element)[0]
        for i in inds_miss:
            print("* index {0:d}: {1}".format(i, str2check[i]))
        if labort:
            raise ValueError("%{0}: Could not find all expected strings in list.".format(method))
    else:
        stat = True
    
    return stat


def check_dir(path2dir: str, lcreate=False):
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


def provide_default(dict_in, keyname, default=None, required=False):
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


