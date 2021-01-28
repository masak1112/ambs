"""
Child class used for configuring the runscript of preprocessing step 2 of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-28"

# import modules
import os, glob
import numpy as np
from data_preprocess.dataset_options import known_datasets
from config_utils import Config_runscript_base    # import parent class

class Config_Preprocess2(Config_runscript_base):

    cls_name = "Config_Preprocess2"#.__name__

    # !!! Important note !!!
    # As long as we don't have runscript templates for all the datasets listed in known_datasets
    # or a generic template runscript, we need the following manual list
    allowed_datasets = ["era5","moving_mnist"]  # known_datasets().keys

    def __init__(self, wrk_flw_step, runscript_base):
        super().__init__(wrk_flw_step, runscript_base)

        #self.sequence_length = None  # only needed for ERA5
                                      # -> do not set to None in order to avoid jeopardizing attribute-check
        # set the attributes which are still None via keyboard interaction
        self.run_config = Config_Preprocess2.run_preprocess2

    def run_preprocess2(self):
        """
        Runs the keyboard interaction for Preprocessing step 2
        :return: all attributes of class Config_Preprocess1 are set
        """
        # decide which dataset is used
        dset_type_req_str = "Enter the name of the dataset for which TFrecords should be prepard for training:\n"
        dset_err = ValueError("Please select a dataset from the ones listed above.")

        self.dataset = Config_Preprocess2.keyboard_interaction(dset_type_req_str, Config_Preprocess2.check_dataset,
                                                               dset_err, ntries=3)

        # get source dir
        source_req_str = "Enter the path where the extracted ERA5 netCDF-files are located:\n"
        source_err = FileNotFoundError("Cannot retrieve extracted ERA5 netCDF-files from passed path.")

        self.source_dir = Config_Preprocess2.keyboard_interaction(source_req_str, Config_Preprocess2.check_data_indir,
                                                                  source_err, ntries=3)
        # check if expected data is available in source_dir (depending on dataset)
        # The following files are expected:
        # * ERA5: pickle-files
        # * moving_MNIST: singgle npy-file
        if self.dataset == "era5":
            if not any(glob.glob(os.path.join(self.source_dir, "**", "*X*.pkl"), recursive=True)):
                raise FileNotFoundError("Could not find any pickle-files under '{0}' which are expected for the "+
                                        "ERA5-dataset.".format(self.source_dir))
        elif self.dataset == "moving_mnist":
            if not os.path.isfile(os.path.join(self.source_dir, "mnist_test_seq.npy")):
                raise FileNotFoundError("Could not find expected file 'mnist_test_seq.npy' under {0}"
                                        .format(self.source_dir))

        if self.dataset == "era5":
            # get desired sequence length
            seql_req_str = "Enter desired total sequence length (i.e. number of frames/images):\n"
            seql_err = ValueError("")

            seql_str = Config_Preprocess2.keyboard_interaction(seql_req_str, Config_Preprocess2.get_seq_length,
                                                               seql_err)
            self.sequence_length = int(seql_str)






    @staticmethod
    # dataset used for training
    def check_dataset(dataset_name, silent=False):
        # NOTE: Templates are only available for ERA5 and moving_MNIST.
        #       After adding further templates or making the template generic,
        #       the latter part of the if-clause can be removed
        #       and further adaptions are required in the configuration chain
        if not dataset_name in Config_Preprocess2.allowed_datasets:
            if not silent:
                print("The following dataset can be used for preproessing step 2:")
                for dataset_avail in Config_Preprocess2.list_datasets: print("* " + dataset_avail)
            return False
        else:
            return True

    @staticmethod
    def check_data_indir(indir, silent=False):
        """
        Check if the passed directory exists. If the required input data is really available depends on the dataset
        and therefore has to be done afterwards
        :param indir: path to input directory from keyboard interaction
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        status = True
        if not os.path.isdir(indir):
            status = False
            if not silent: print("Could not find data directory '{0}'.".format(indir))

        return status

    @staticmethod
    def get_seq_length(seq_length, silent=False):
        """
        Check if passed sequence length is larger than 1 (lower limit for meaningful prediction)
        :param seq_length: sequence length from keyboard interaction
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        status = False
        if seq_length.strip().isnumeric():
            if int(seq_length) >= 2:
                status = True

        return status



