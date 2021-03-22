"""
Some auxilary routines which may are used throughout the project.
Provides:   * get_unique_vars
            *

"""

# import modules
import os
import sys
import numpy as np

# routines
def get_unique_vars(varnames):
    """
    :param varnames: list of variable names (or any other list of strings)
    :return: list with unique elements of inputted varnames list
    """
    vars_uni, varsind = np.unique(varnames, return_index=True)
    nvars_uni = len(vars_uni)

    return (vars_uni, varsind, nvars_uni)


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

        line_str = line_str + add_str + "/"
    else:
        print(add_str + " is already part of " + line_str + ". No change is performed.")

    if l_linebreak:  # re-add carriage return to string if required
        return (line_str + "\n")
    else:
        return (line_str)


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

    stat = False
    if isinstance(str2check, str):
        str2check = [str2check]
    elif isinstance(str2check, list):
        assert np.all([isinstance(str1, str) for str1 in str2check]) == True, \
            "Not all elements of str2check are strings"
    else:
        raise ValueError("str2check argument must be either a string or a list of strings")

    stat_element = [True if str1 in list_in else False for str1 in str2check]

    if not np.all(stat_element):
        print("The following elements are not part of the input list:")
        inds_miss = np.where(stat_element)[0]
        for i in inds_miss:
            print("* index {0:d}: {1}".format(i, str2check[i]))
        if labort:
            raise ValueError("Could not find all expected strings in list.")
    else:
        stat = True
    
    return stat
