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
sys.path.append(path.abspath('../model_modules/'))
from model_architectures import known_models

# start script

def get_variable_from_runscript(runscript_file,script_variable):
    '''
    Searach for the declaration of variable in a Shell script and returns its value.
    '''
    script_variable = script_variable + "="
    found = False

    with open(runscript_file) as runscript:
        # Skips text before the beginning of the interesting block:
        for line in runscript:
            if script_variable in line:
                var_value = line.strip(script_variable)
                found = True
                break

    if not found:
        raise Exception("Could not find declaration of '"+script_variable+"' in '"+runscript_file+"'.")

    return var_value


def main():

    known_models = known_models().keys()

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
    model = input("Enter the name of the model you want to train:\n")

    # sanity check (is the model implemented?)
    if not (model in known_models):
        print("The following models are implemented in the workflow:")
        for model_avail in known_models: print("* "+model_avail)
        raise ValueError("Could not find the passed model '"+model+"'! Please select a model from the ones listed above.")

    # experimental ID
    exp_id = input("Enter your desired experimental id:\n")

    # also get current timestamp and user-name
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    user_name = os.environ["USER"]

    target_dir = timestamp +"_"+ user_name +"_"+ exp_id
    base_dir = get_variable_from_runscript('train_model_era5_template.sh','source_dir')
    target_dir = os.path.join(base_dir,target_dir)

    # sanity check (target_dir is unique):
    if os.path.isdir(target_dir):
        raise IsADirectoryError(target_dir+" already exists! Make sure that it is unique.")



















