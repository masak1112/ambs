"""
Child class used for configuring the training runscript of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-29"

# import modules
import os, glob
import time
import datetime as dt
import subprocess as sp
from model_modules.model_architectures import known_models
from data_preprocess.dataset_options import known_datasets
from config_utils import Config_runscript_base    # import parent class

class Config_Train(Config_runscript_base):
    cls_name = "Config_Train"#.__name__

    list_models = known_models().keys()
    # !!! Important note !!!
    # As long as we don't have runscript templates for all the datasets listed in known_datasets
    # or a generic template runscript, we need the following manual list
    allowed_datasets = ["era5","moving_mnist"]  # known_datasets().keys

    def __init__(self, venv_name, lhpc):
        super().__init__(venv_name, lhpc)

        # initialize attributes related to runscript name
        self.long_name_wrk_step = "Training"
        self.rscrpt_tmpl_prefix = "train_model_"
        # initialize additional runscript-specific attributes to be set via keyboard interaction
        self.model = None
        self.destination_dir = None
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "model", "exp_id", "destination_dir"]
        # copy over method for keyboard interaction
        self.run_config = Config_Train.run_training
    #
    # -----------------------------------------------------------------------------------
    #
    def run_training(self):
        """
        Runs the keyboard interaction for Training
        :return: all attributes of class training are set
        """

        # decide which dataset is used
        dset_type_req_str = "Enter the name of the dataset on which you want to train:\n"
        dset_err = ValueError("Please select a dataset from the ones listed above.")

        self.dataset = Config_Train.keyboard_interaction(dset_type_req_str, Config_Train.check_dataset,
                                                         dset_err, ntries=2)

        # now, we are also ready to set the correct name of the runscript template and the target
        self.runscript_template = self.rscrpt_tmpl_prefix + self.dataset + self.suffix_template
        self.runscript_target = self.rscrpt_tmpl_prefix + self.dataset + ".sh"

        # get the source directory

        expdir_req_str = "Enter the path to the preprocessed data (directory where tf-records files are located):\n"
        expdir_err = FileNotFoundError("Could not find any tfrecords.")

        self.source_dir = Config_Train.keyboard_interaction(expdir_req_str, Config_Train.check_expdir,
                                                            expdir_err, ntries=3)

        # split up directory path in order to retrieve exp_dir used for setting up the destination directory
        exp_dir_split = path_rec_split(self.source_dir)
        index = [idx for idx, s in enumerate(exp_dir_split) if self.dataset in s]
        if index == []:
            raise ValueError(
                "tfrecords found under '{0}', but directory does not seem to reflect naming convention.".format(
                    self.source_dir))
        exp_dir = exp_dir_split[index[0]]

        # get the model to train
        model_req_str = "Enter the name of the model you want to train:\n"
        model_err     = ValueError("Please select a model from the ones listed above.")

        self.model = Config_Train.keyboard_interaction(model_req_str, Config_Train.check_model, model_err, ntries=2)

        # experimental ID
        # No need to call keyboard_interaction here, because the user can pass whatever we wants
        self.exp_id = input("Enter your desired experimental id (will be extended by timestamp and username):\n")

        # also get current timestamp and user-name...
        timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        user_name = os.environ["USER"]
        # ... to construct final destination_dir and exp_dir_ext as well
        self.exp_id = timestamp +"_"+ user_name +"_"+ self.exp_id  # by convention, exp_id is extended by timestamp and username

        base_dir   = get_variable_from_runscript("train_model_{0}_template.sh".format(self.dataset), "destination_dir")
        exp_dir_ext= os.path.join(exp_dir, self.model, self.exp_id)
        self.destination_dir = os.path.join(base_dir, exp_dir, self.model, self.exp_id)

        # sanity check (target_dir is unique):
        if os.path.isdir(self.destination_dir):
            raise IsADirectoryError(self.destination_dir+" already exists! Make sure that it is unique.")

        # create destination directory...
        os.makedirs(self.destination_dir)

        # Create json-file for data splitting
        source_datasplit = os.path.join("..", "data_split", self.dataset, self.model, "datasplit.json")
        # sanity check (default data_split json-file exists)
        if not os.path.isfile(source_datasplit):
            raise FileNotFoundError("Could not find default data_split json-file '"+source_datasplit+"'")
        # ...copy over json-file for data splitting...
        os.system("cp "+source_datasplit+" "+self.destination_dir)
        # ...and open vim after some delay
        print("You may now configure the data splitting:\n")
        time.sleep(3)
        cmd_vim = os.environ.get('EDITOR', 'vi') + ' ' + os.path.join(self.destination_dir,"data_split.json")
        sp.call(cmd_vim, shell=True)

        # Create json-file for hyperparameters
        source_hparams = os.path.join("..","hparams", self.dataset, self.model, "model_hparams.json")
        # sanity check (default hyperparameter json-file exists)
        if not os.path.isfile(source_hparams):
            raise FileNotFoundError("Could not find default hyperparameter json-file '"+source_hparams+"'")
        # ...copy over json-file for hyperparamters...
        os.system("cp "+source_hparams+" "+self.destination_dir)
        # ...and open vim after some delay
        print("You may now configure the model hyperparameters:\n")
        time.sleep(3)
        cmd_vim = os.environ.get('EDITOR', 'vi') + ' ' + os.path.join(self.destination_dir, "model_hparams.json")
        sp.call(cmd_vim, shell=True)
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    # dataset used for training
    def check_dataset(dataset_name, silent=False):
        """
        Check if the passed dataset name is known
        :param dataset_name: dataset from keyboard interaction
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        # NOTE: Templates are only available for ERA5 and moving_MNIST.
        #       After adding further templates or making the template generic,
        #       the latter part of the if-clause can be removed
        #       and further adaptions are required in the configuration chain
        if not dataset_name in Config_Train.allowed_datasets:
            if not silent:
                print("The following dataset can be used for training:")
                for dataset_avail in Config_Train.allowed_datasets: print("* " + dataset_avail)
            return False
        else:
            return True
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_expdir(exp_dir, silent=False):
        """
        Check if the passed directory path contains TFrecord-files. Note, that the path is extended by tfrecords/
        (e.g. see <base_dir>/model_modules/video_prediction/datasets/era5_dataset.py)
        :param exp_dir: path from keyboard interaction
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        status = False
        real_dir = os.path.join(exp_dir, "tfrecords")
        if os.path.isdir(real_dir):
            file_list = glob.glob(os.path.join(real_dir, "sequence*.tfrecords"))
            if len(file_list) > 0:
                status = True
            else:
                print("{0} does not contain any tfrecord-files.".format(real_dir))
        else:
            if not silent: print("Passed directory does not exist!")
        return status
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_model(model_name, silent=False):
        """
        Check if the passed model name is known/available.
        :param model_name: model name from keyboard interaction
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        if not (model_name in Config_Train.list_models):
            if not silent:
                print("{0} is not a valid model!".format(model_name))
                print("The following models are implemented in the workflow:")
                for model_avail in Config_Train.list_models: print("* " + model_avail)
            return False
        else:
            return True
#
# -----------------------------------------------------------------------------------
#