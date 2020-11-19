"""
Basic task of the Python-script:

Creates user-defined runscripts for training, set ups a user-defined target directory and allows for full control
on the setting of hyperparameters.
"""

__email__ = "b.gong@fz-juelich.de"
__authors__ = "Michael Langguth"
__date__ = "2020-11-19"

# import modules
import sys, os, glob
import subprocess
import datetime as dt
import json as js
from os import path
if sys.version_info[0] < 3:
    raise Exception("This script has to be run with Python 3!")
sys.path.append(os.path.dirname(sys.path[0]))
from model_modules.model_architectures import known_models
from data_preprocess.dataset_options import known_datasets

# some auxiliary functions

# robust check if script is running in virtual env from
# https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv/38939054
def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix
#
#--------------------------------------------------------------------------------------------------------
#
def in_virtualenv():
    return get_base_prefix_compat() != sys.prefix
#
#--------------------------------------------------------------------------------------------------------
#
def check_virtualenv(labort=False):
    '''
    Checks if current script is running a virtual environment and returns the directory's name
    :param labort: If True, the an Exception is raised. If False, only a Warning is given
    :return: name of virtual environment
    '''
    lvirt = in_virtualenv()

    if not lvirt:
        if labort:
            raise EnvironmentError("config_train.py has to run in an activated virtual environment!")
        else:
            raise Warning("config_train.py is not running in an activated virtual environment!")
            return
    else:
        return os.path.basename(sys.prefix)
#
# --------------------------------------------------------------------------------------------------------
#
def get_variable_from_runscript(runscript_file,script_variable):
    '''
    Search for the declaration of variable in a Shell script and returns its value.
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
                var_value = (line.strip(script_variable)).replace("\n", "")
                found = True
                break

    if not found:
        raise Exception("Could not find declaration of '"+script_variable+"' in '"+runscript_file+"'.")

    return var_value
#
#--------------------------------------------------------------------------------------------------------
#
def path_rec_split(full_path):
    """
    :param full_path: input path to be splitted in its components
    :return: list of all splitted components
    """
    rest, tail = os.path.split(full_path)
    if rest in ('', os.path.sep): return tail,

    return path_rec_split(rest) + (tail,)
#
#--------------------------------------------------------------------------------------------------------
#
def keyboard_interaction(console_str,check_input,err,ntries=1):
    """
    Function to check if the user has passed a proper input via keyboard interaction
    :param console_str: Request printed to the console
    :param check_input: function returning boolean which needs to be passed by input from keyboard interaction.
                        Must have two arguments with the latter being an optional bool called silent.
    :param ntries: maximum number of tries (default: 1)
    :return: The approved input from keyboard interaction
    """
    # sanity checks
    if not callable(check_input):
        raise ValueError("check_input must be a function!")
    else:
        try:
            if not type(check_input("xxx",silent=True)) is bool:
                raise TypeError("check_input argument does not return a boolean.")
            else:
                pass
        except:
            raise Exception("Cannot approve check_input-argument to be proper.")
    if not isinstance(err,BaseException):
        raise ValueError("err_str-argument must be an instance of BaseException!")
    if not isinstance(ntries,int) and ntries <= 1:
        raise ValueError("ntries-argument must be an integer greater equal 1!")

    attempt = 0
    while attempt < ntries:
        input_req = input(console_str)
        if check_input(input_req):
            break
        else:
            attempt += 1
            if attempt < ntries:
                print(err)
                console_str = "Retry!\n"
            else:
                raise err

    return input_req


def main():

    list_models = known_models().keys()
    list_datasets = known_datasets().keys()

    # sanity check (is Python running in a virtual environment)
    venv_name = check_virtualenv(labort=True)

    ## get required information from the user by keyboard interaction

    # dataset used for training
    def check_dataset(dataset_name, silent=False):
        # NOTE: Generic template for training still has to be integrated!
        #       After this is done, the latter part of the if-clause can be removed
        #       and further adaptions for the target_dir and for retrieving base_dir (see below) are required
        if not dataset_name in list_datasets or dataset_name != "era5":
            if not silent:
                print("The following dataset can be used for training:")
                for dataset_avail in list_datasets: print("* " + dataset_avail)
            return False
        else:
            return True

    dataset_req_str = "Enter the name of the dataset for training:\n"
    dataset_err     = ValueError("Please select a dataset from the ones listed above.")

    dataset = keyboard_interaction(dataset_req_str,check_dataset,dataset_err,ntries=2)
    # path to preprocessed data
    def check_expdir(exp_dir, silent=False):
        status = False
        if os.path.isdir(exp_dir):
            file_list = glob.glob(os.path.join(exp_dir,"sequence*.tfrecords"))
            if len(file_list) > 0:
                status = True
            else:
                print("{0} does not contain any tfrecord-files.".format(exp_dir))
        else:
            if not silent: print("Passed directory does not exist!")
        return status

    expdir_req_str = "Enter the path to the preprocessed data (directory where tf-records files are located):\n"
    expdir_err     = FileNotFoundError("Could not find any tfrecords.")

    exp_dir_full   = keyboard_interaction(expdir_req_str, check_expdir, expdir_err, ntries=3)

    # split up directory path
    exp_dir_split = path_rec_split(exp_dir_full)
    index = [idx for idx, s in enumerate(exp_dir_split) if dataset in s]
    if index == []:
        raise ValueError("tfrecords found under '{0}', but directory does not seem to reflect naming convention.".format(exp_dir_full))
    exp_dir = exp_dir_split[index[0]]

    # model
    def check_model(model_name, silent=False):
        if not model_name in list_models:
            if not silent:
                print("{0} is not a valid model!".format(model_name))
                print("The following models are implemented in the workflow:")
                for model_avail in list_models: print("* " + model_avail)
            return False
        else:
            return True

    model_req_str = "Enter the name of the model you want to train:\n"
    model_err     = ValueError("Please select a model from the ones listed above.")

    model = keyboard_interaction(model_req_str, check_model, model_err, ntries=2)

    # experimental ID
    # No need to call keyboard_interaction here, because the user can pass whatever we wants
    exp_id = input("Enter your desired experimental id (will be extended by timestamp and username):\n")

    # also get current timestamp and user-name...
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    user_name = os.environ["USER"]
    # ... to construct final target_dir and exp_dir_ext as well
    exp_id = timestamp +"_"+ user_name +"_"+ exp_id  # by convention, exp_id is extended by timestamp and username
    base_dir   = get_variable_from_runscript('train_model_era5_template.sh','destination_dir')
    exp_dir_ext= os.path.join(exp_dir,model,exp_id)
    target_dir = os.path.join(base_dir,exp_dir,model,exp_id)

    # sanity check (target_dir is unique):
    if os.path.isdir(target_dir):
        raise IsADirectoryError(target_dir+" already exists! Make sure that it is unique.")

    # create destination directory...
    os.makedirs(target_dir)
    source_hparams = os.path.join("..","hparams",dataset,model,"model_hparams.json")
    # sanity check (default hyperparameter json-file exists)
    if not os.path.isfile(source_hparams):
        raise FileNotFoundError("Could not find default hyperparameter json-file '"+source_hparams+"'")
    # ...copy over json-file for hyperparamters...
    os.system("cp "+source_hparams+" "+target_dir)
    # ...and open vim
    cmd_vim = os.environ.get('EDITOR', 'vi') + ' ' + os.path.join(target_dir,"model_hparams.json")
    subprocess.call(cmd_vim, shell=True)

    # finally, create runscript for training...
    cmd = "cd ../env_setup; ./generate_workflow_runscripts.sh ../HPC_scripts/train_model_era5 "+ venv_name+ \
          " -exp_id="+exp_id+" -exp_dir="+exp_dir+" -exp_dir_ext="+exp_dir_ext+" -model="+model+" ; cd -"
    os.system(cmd)
    # ...and postprocessing as well
    cmd = cmd.replace("train_model_era5","visualize_postprocess_era5")
    os.system(cmd)

if __name__== '__main__':
    main()















