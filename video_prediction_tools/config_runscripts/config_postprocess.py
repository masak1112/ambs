"""
Child class used for configuring the postprocessing runscript of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-02-01"

# import modules
import os, glob
from model_modules.model_architectures import known_models
from data_preprocess.dataset_options import known_datasets
from config_utils import Config_runscript_base    # import parent class

class Config_Postprocess(Config_runscript_base):
    cls_name = "Config_Postprocess"#.__name__

    list_models = known_models().keys()
    # !!! Important note !!!
    # As long as we don't have runscript templates for all the datasets listed in known_datasets
    # or a generic template runscript, we need the following manual list
    allowed_datasets = ["era5","moving_mnist"]  # known_datasets().keys

    def __init__(self, venv_name, lhpc):
        super().__init__(venv_name, lhpc)

        # initialize attributes related to runscript name
        self.long_name_wrk_step = "Postprocessing"
        self.rscrpt_tmpl_prefix = "visualize_postprocess_"
        # initialize additional runscript-specific attributes to be set via keyboard interaction
        self.model = None
        self.checkpoint_dir = None
        self.destination_dir = None
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "results_dir",
                                "checkpoint_dir", "model"]
        # copy over method for keyboard interaction
        self.run_config = Config_Postprocess.run_postprocess
    #
    # -----------------------------------------------------------------------------------
    #
    def run_postprocess(self):
        """
        Runs the keyboard interaction for Postprocessing
        :return: all attributes of class postprocess are set
        """
        # decide which dataset is used
        dset_type_req_str = "Enter the name of the dataset on which training was performed:"
        dset_err = ValueError("Please select a dataset from the ones listed above.")

        self.dataset = Config_Postprocess.keyboard_interaction(dset_type_req_str, Config_Postprocess.check_dataset,
                                                              dset_err, ntries=2)

        # now, we are also ready to set the correct name of the runscript template and the target
        self.runscript_template = self.rscrpt_tmpl_prefix + self.dataset + self.suffix_template
        # sel.runscript_target is set below

        # get the 'checkpoint-directory', i.e. the directory where the trained model parameters are stored
        # Note that the remaining information (model, results-directory etc.) can be retrieved form it!!!
        trained_dir_req_str = "Enter the absolute (!) path to the model checkpoint directory" + \
                              " for which postprocessing should be done:"
        trained_err = FileNotFoundError("No trained model parameters found.")

        self.checkpoint_dir = Config_Postprocess.keyboard_interaction(trained_dir_req_str,
                                                                      Config_Postprocess.check_traindir,
                                                                      trained_err, ntries=3)

        # get the relevant information from checkpoint_dir in order to construct source_dir and results_dir
        # (following naming convention)
        cp_dir_split = Config_Postprocess.path_rec_split(self.checkpoint_dir)
        cp_dir_split = list(filter(None, cp_dir_split))                       # get rid of empty list elements

        base_dir, exp_dir_base, exp_dir = "/"+os.path.join(*cp_dir_split[:-4]), cp_dir_split[-3], cp_dir_split[-1]
        self.model = Config_Postprocess.check_model(cp_dir_split[-2])
        self.runscript_target = self.rscrpt_tmpl_prefix + self.dataset + exp_dir + ".sh"

        self.source_dir = Config_Postprocess.check_source(os.path.join(base_dir, "preprocessedData", exp_dir_base))
        self.destination_dir = os.path.join(base_dir, "results", exp_dir_base, self.model, exp_dir)
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    # dataset used for postprocessing
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
        if not dataset_name in Config_Postprocess.allowed_datasets:
            if not silent:
                print("The following dataset can be used for training:")
                for dataset_avail in Config_Postprocess.allowed_datasets: print("* " + dataset_avail)
            return False
        else:
            return True
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_traindir(checkpoint_dir, silent=False):
        """
        Check if the passed directory path contains the model parameters stored in model-XXXX.meta
        :param checkpoint_dir: path from keyboard interaction
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        status = False
        if os.path.isdir(checkpoint_dir):
            file_list = glob.glob(os.path.join(checkpoint_dir, "model*.meta"))
            if len(file_list) > 0:
                status = True
            else:
                print("{0} does not contain any model parameter files (model-*.meta).".format(checkpoint_dir))
        else:
            if not silent: print("Passed directory does not exist!")
        return status
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_model(model_in):
        """
        Checks if passed model name is known and can be handled
        :param model_in: name of model to be checked
        :return: returns model_in when check is passed successfully
        """
        if not model_in in Config_Postprocess.list_models:
            print("**** Known models ****")
            for model in Config_Postprocess.list_models: print(model)
            raise ValueError("{0} is an unknown model (see list of known models above).".format(model_in))
        else:
            pass

        return model_in
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_source(source_dir_in):
        """
        Checks if TFrecord files (used for training) are available. These are used to construct the test-dataset
        :param source_dir_in: input directory to be checked
        :return: returns source_dir_in when check is passed successfully
        """
        real_dir = os.path.join(source_dir_in, "tfrecords")
        if os.path.isdir(real_dir):
            file_list = glob.glob(os.path.join(real_dir, "sequence*.tfrecords"))
            if len(file_list) > 0:
                pass
            else:
                raise FileNotFoundError("{0} does not contain any tfrecord-files.".format(real_dir))
        else:
            raise NotADirectoryError("Cannot find directory '{0}'.".format(real_dir))

        return source_dir_in


