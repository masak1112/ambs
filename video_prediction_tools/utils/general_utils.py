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