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
#import numpy as np
import datetime as dt
import json as js
from os import path
sys.path.append(path.abspath('../utils/'))
import metadata
sys.path.append(path.abspath('../'))
from model_modules.model_architectures import known_models
from data_preprocess.dataset_options import known_datasets

# start script

def get_variable_from_runscript(runscript_file,script_variable):
    '''
    Searach for the declaration of variable in a Shell script and returns its value.
    :param runscript_file: path to shell script/runscript
    :param script_variable: name of variable which is declared in shell script at hand
    :return: value of script_variable
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

def path_rec_split(full_path):
    """
    :param full_path: input path to be splitted in its components
    :return: list of all splitted components
    """
    rest, tail = os.path.split(full_path)
    if rest == '':
        return tail,
    return rec_split(rest) + (tail,)


def main():

    list_models = known_models().keys()
    list_datasets = known_datasets().keys()

    # get required information from the user by keyboard interaction

    # dataset used for training
    dataset = input("Enter the name of the dataset for training:\n")

    # NOTE: Generic template for training still has to be integrated!
    #       After this is done, the latter part of the if-clause can be removed
    #       and further adaptions for the target_dir and for retrieving base_dir (see below) are required
    if not dataset in list_datasets and dataset == "era5":
        print("The following dataset can be used for training:")
        for dataset_avail in list_datasets: print("* "+dataset_avail)
        raise ValueError("Please select a dataset from the ones listed above.")


    # path to preprocessed data
    exp_dir_full = input("Enter the path to the preprocessed data (directory where tf-records files are located):\n")
    exp_dir_full = os.path.join(exp_dir_full,"train")
    # sanity check (does preprocessed data exist?)
    if not (os.path.isdir(exp_dir_full)):
        raise NotADirectoryError("Passed path to preprocessed data '"+exp_dir_full+"' does not exist!")
    file_list = glob.glob(os.path.join(exp_dir_full,"sequence*.tfrecords"))
    if len(file_list) == 0:
        raise FileNotFoundError("Passed path to preprocessed data '"+exp_dir_full+"' exists,"+\
                                "but no tfrecord-files can be found therein")

    exp_dir_split = path_rec_split(exp_dir_full)
    index = [idx for idx, s in enumerate(l) if dataset in s][0]
    exp_dir = exp_dir_split[index]

    # path to virtual environment to be used
    venv_name = input("Enter the name of the virtual environment which should be used:\n")

    # sanity check (does virtual environment exist?)
    if not (os.path.isfile(os.path.join("../",venv_name,"bin","activate"))):
        raise FileNotFoundError("Could not find a virtual environment named "+venv_name)

    # model
    model = input("Enter the name of the model you want to train:\n")

    # sanity check (is the model implemented?)
    if not (model in list_models):
        print("The following models are implemented in the workflow:")
        for model_avail in list_models: print("* "+model_avail)
        raise ValueError("Could not find the passed model '"+model+"'! Please select a model from the ones listed above.")

    # experimental ID
    exp_id = input("Enter your desired experimental id:\n")

    # also get current timestamp and user-name
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    user_name = os.environ["USER"]

    target_dir = timestamp +"_"+ user_name +"_"+ exp_id
    base_dir = get_variable_from_runscript('train_model_era5_template.sh','destination_dir')
    target_dir = os.path.join(base_dir,target_dir)

    # sanity check (target_dir is unique):
    if os.path.isdir(target_dir):
        raise IsADirectoryError(target_dir+" already exists! Make sure that it is unique.")

    print(target_dir)

if __name__== '__main__':
    main()















