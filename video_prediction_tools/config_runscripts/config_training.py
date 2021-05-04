"""
Child class used for configuring the training runscript of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-29"

# import modules
import os, glob
import re
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
    allowed_datasets = ["era5", "moving_mnist"]  # known_datasets().keys

    basename_tfdirs = "tfrecords_seq_len_"

    def __init__(self, venv_name, lhpc):
        super().__init__(venv_name, lhpc)

        # initialize attributes related to runscript name
        self.long_name_wrk_step = "Training"
        self.rscrpt_tmpl_prefix = "train_model_"
        # initialize additional runscript-specific attributes to be set via keyboard interaction
        self.model = None
        self.destination_dir = None
        self.datasplit_dict = None
        self.model_hparams = None
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "model", "destination_dir", "datasplit_dict",
                                "model_hparams"]
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
        method_name = Config_Train.run_training.__name__

        # decide which dataset is used
        dset_type_req_str = "Enter the name of the dataset on which you want to train:"
        dset_err = ValueError("Please select a dataset from the ones listed above.")

        self.dataset = Config_Train.keyboard_interaction(dset_type_req_str, Config_Train.check_dataset,
                                                         dset_err, ntries=2)

        # get source dir (relative to base_dir_source!)
        self.runscript_template = os.path.join(self.runscript_dir, "train_model_{0}{1}".format(self.dataset, self.suffix_template))
        source_dir_base = Config_Train.handle_source_dir(self, "preprocessedData")

        expdir_req_str = "Choose a subdirectory listed above where the preprocessed TFrecords are located:"
        expdir_err = FileNotFoundError("Could not find any tfrecords.")

        self.source_dir = Config_Train.keyboard_interaction(expdir_req_str, Config_Train.check_expdir,
                                                            expdir_err, ntries=3, prefix2arg=source_dir_base+"/")
        # expand source_dir by tfrecords-subdirectory
        tf_dirs = Config_Train.list_tf_dirs(self)
        ntf_dirs = len(tf_dirs)

        if ntf_dirs == 1:
            self.source_dir = os.path.join(self.source_dir, tf_dirs[0])
        else:
            seq_req_str = "Enter desired sequence length of TFrecord-files to be used for training " + \
                          "(see suffix from list above)"
            seq_err = FileNotFoundError("Invalid sequence length passed or no TFrecord-files found")

            # Note, how the check_expdir-method is recycled by simply adding a properly defined suffix to it
            # Note that due to the suffix, the returned value already corresponds to the final path for source_dir
            self.source_dir = Config_Train.keyboard_interaction(seq_req_str, Config_Train.check_expdir,
                                                                seq_err, ntries=2,
                                                                prefix2arg=os.path.join(self.source_dir,
                                                                                        Config_Train.basename_tfdirs))


        # split up directory path in order to retrieve exp_dir used for setting up the destination directory
        exp_dir_split = Config_Train.path_rec_split(self.source_dir)
        index = [idx for idx, s in enumerate(exp_dir_split) if self.dataset in s]
        if not index:
            raise ValueError(
                    "%{0}: tfrecords found under '{1}', but directory does not seem to reflect naming convention."
                    .format(method_name, self.source_dir))
        exp_dir = exp_dir_split[index[0]]

        # get the model to train
        model_req_str = "Enter the name of the model you want to train:"
        model_err = ValueError("Please select a model from the ones listed above.")

        self.model = Config_Train.keyboard_interaction(model_req_str, Config_Train.check_model, model_err, ntries=2)

        # experimental ID
        # No need to call keyboard_interaction here, because the user can pass whatever we wants
        self.exp_id = input("*** Enter your desired experimental id (will be extended by timestamp and username):\n")

        # also get current timestamp and user-name...
        timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        user_name = os.environ["USER"]
        # ... to construct final destination_dir and exp_dir_ext as well
        self.exp_id = timestamp + "_" + user_name + "_" + self.exp_id  # by convention, exp_id is extended by timestamp and username

        # now, we are also ready to set the correct name of the runscript template and the target
        self.runscript_target = "{0}{1}_{2}.sh".format(self.rscrpt_tmpl_prefix, self.dataset, self.exp_id)
        
        base_dir = Config_Train.get_var_from_runscript(os.path.join(self.runscript_dir, self.runscript_template),
                                                         "destination_dir")
        exp_dir_ext = os.path.join(exp_dir, self.model, self.exp_id)
        self.destination_dir = os.path.join(base_dir, "models", exp_dir, self.model, self.exp_id)
        
        # sanity check (target_dir is unique):
        if os.path.isdir(self.destination_dir):
            raise IsADirectoryError("%{0}: {1} already exists! Make sure that it is unique."
                                    .format(method_name, self.destination_dir))

        # create destination directory...
        os.makedirs(self.destination_dir)

        # Create json-file for data splitting
        source_datasplit = os.path.join("..", "data_split", self.dataset, "datasplit_template.json")
        self.datasplit_dict = os.path.join(self.destination_dir, "data_split.json")
        # sanity check (default data_split json-file exists)
        if not os.path.isfile(source_datasplit):
            raise FileNotFoundError("%{0}: Could not find default data_split json-file '{1}'".format(method_name,
                                                                                                     source_datasplit))
        # ...copy over json-file for data splitting...
        os.system("cp "+source_datasplit+" "+self.datasplit_dict)
        # ...and open vim after some delay
        print("*** Please configure the data splitting:")
        time.sleep(3)
        cmd_vim = os.environ.get('EDITOR', 'vi') + ' ' + os.path.join(self.destination_dir,"data_split.json")
        sp.call(cmd_vim, shell=True)
        sp.call("sed -i '/^#/d' {0}".format(self.datasplit_dict), shell=True)

        # Create json-file for hyperparameters
        source_hparams = os.path.join("..","hparams", self.dataset, self.model, "model_hparams_template.json")
        self.model_hparams = os.path.join(self.destination_dir, "model_hparams.json")
        # sanity check (default hyperparameter json-file exists)
        if not os.path.isfile(source_hparams):
            raise FileNotFoundError("%{0}: Could not find default hyperparameter json-file '%{1}'"
                                    .format(method_name, source_hparams))
        # ...copy over json-file for hyperparamters...
        os.system("cp "+source_hparams+" "+self.model_hparams)
        # ...and open vim after some delay
        print("*** Please configure the model hyperparameters:")
        time.sleep(3)
        cmd_vim = os.environ.get('EDITOR', 'vi') + ' ' + self.model_hparams
        sp.call(cmd_vim, shell=True)
    #
    # -----------------------------------------------------------------------------------
    #
    def list_tf_dirs(self):

        method = Config_Train.list_tf_dirs.__name__
        if self.source_dir is None:
            raise AttributeError("%{0}: source_dir must be non-emoty.".format(method))

        all_dirs = [f.name for f in os.scandir(self.source_dir) if f.is_dir()]
        tf_dirs = [dirname for dirname in all_dirs if dirname.startswith("tfrecords_seq_len")]

        if not tf_dirs:
            raise ValueError("%{0}: Could not find properly named TFrecords-directory under '{0}'"
                             .format(self.source_dir))
        ndirs = len(tf_dirs)

        if ndirs == 1:
            print("Found the following directory for TFrecord-files: '{0}'".format(tf_dirs[0]))
        else:
            print("The following directories for TFrecords-files have been found:")
            for idir in tf_dirs:
                print("* {0}".format(idir))

        return tf_dirs


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
        Check if the passed directory path contains at least on subdirectory where TFrecord-files are located.
        :param exp_dir: path from keyboard interaction
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        status = False
        if os.path.isdir(exp_dir):
            if re.match(".+{0}".format(Config_Train.basename_tfdirs), exp_dir):
                status = any(glob.iglob(os.path.join(exp_dir, "sequence*.tfrecords")))
            else:
                status = any(glob.iglob(os.path.join(exp_dir, "*", "sequence*.tfrecords")))
            if not status and not silent:
                print("{0} does not contain any tfrecord-files.".format(exp_dir))
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
