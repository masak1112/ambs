"""
Basic task of the Python-script:

Creates user-defined runscripts for training, set ups a user-defined target directory and allows for full control
on the setting of hyperparameters.
"""

__email__ = "b.gong@fz-juelich.de"
__authors__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"
__date__ = "2020-10-27"

# import modules
import sys, os, glob
import numpy as np
import datetime as dt
import json as js
import metadata
sys.path.append(path.abspath('../video_prediction/'))
from models import get_model_class

known_architectures = ["savp","convLSTM","vae","mcnet"]
if not (model in known_architectures):


# start script

def main():
    # get required information from the user by keyboard interaction

    # path to preprocessed data
    exp_dir = input("Enter the path to the preprocessed data (directory where tf-records files are located):\n")
    exp_dir = os.path.join(exp_dir,"train")
    # sanity check (does preprocessed data exist?)
    if not (os.path.isdir(exp_dir)):
        raise NotADirectoryError("Passed path to preprocessed data '"+exp_dir+"' does not exist!")
    file_list = glob.glob(os.path.join(exp_dir,"sequence*.tfrecords"))
    if len(file_list) == 0:
        raise FileNotFoundError("Passed path to preprocessed data '"+exp_dir+"' exists,"+\
                                "but no tfrecord-files can be found therein")

    # path to virtual environment to be used
    venv_name = input("Enter the name of the virtual environment which should be used:\n")

    # sanity check (does virtual environment exist?)
    if not (os.path.isfile("../",venv_name,"bin","activate")):
        raise FileNotFoundError("Could not find a virtual environment named "+venv_name)

    # model


    # experimental ID
    exp_id = input("Enter your desired experimental id:\n")

    # also get current timestamp and user-name
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    user_name = os.environ["USER"]











