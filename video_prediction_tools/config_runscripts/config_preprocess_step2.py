"""
Child class used for configuring the preprocessing step 2 runscript of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-28"

# import modules
import os, glob
from data_preprocess.dataset_options import known_datasets
from config_utils import Config_runscript_base    # import parent class

class Config_Preprocess2(Config_runscript_base):

    cls_name = "Config_Preprocess2"#.__name__

    # !!! Important note !!!
    # As long as we don't have runscript templates for all the datasets listed in known_datasets
    # or a generic template runscript, we need the following manual list
    allowed_datasets = ["era5","moving_mnist"]  # known_datasets().keys

    def __init__(self, venv_name, lhpc):
        super().__init__(venv_name, lhpc)

        # initialize attributes related to runscript name
        self.long_name_wrk_step = "Preproccessing step 2"
        self.rscrpt_tmpl_prefix = "preprocess_data_"
        # initialize additional runscript-specific attributes to be set via keyboard interaction
        self.destination_dir = None
        self.sequence_length = None  # only needed for ERA5
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "destination_dir"]     # appended for ERA5 dataset
        # copy over method for keyboard interaction
        self.run_config = Config_Preprocess2.run_preprocess2
    #
    # -----------------------------------------------------------------------------------
    #
    def run_preprocess2(self):
        """
        Runs the keyboard interaction for Preprocessing step 2
        :return: all attributes of class Config_Preprocess2 set
        """
        # decide which dataset is used
        dset_type_req_str = "Enter the name of the dataset for which TFrecords should be prepard for training:\n"
        dset_err = ValueError("Please select a dataset from the ones listed above.")

        self.dataset = Config_Preprocess2.keyboard_interaction(dset_type_req_str, Config_Preprocess2.check_dataset,
                                                               dset_err, ntries=3)
        # now, we are also ready to set the correct name of the runscript template and the target
        self.runscript_template = self.rscrpt_tmpl_prefix + self.dataset + "_step2"+\
                                  self.suffix_template
        self.runscript_target = self.rscrpt_tmpl_prefix + self.dataset + "_step2" + ".sh"

        # get source dir (relative to base_dir_source!)
        source_dir_base = Config_Preprocess2.handle_source_dir(self, "preprocessedData")

        if self.dataset == "era5":
            file_type = "ERA5 pickle-files are"
        elif self.dataset == "moving_mnist":
            file_type = "The movingMNIST data file is"
        source_req_str = "Choose a subdirectory listed above to {0} where the extracted {1} located:\n"\
                             .format(base_dir_source, file_type)
        source_err = FileNotFoundError("Cannot retrieve "+file_type+" from passed path.")

        self.source_dir = Config_Preprocess2.keyboard_interaction(source_req_str, Config_Preprocess2.check_data_indir,
                                                                  source_err, ntries=3, suffix2arg=source_dir_base+"/")
        # Note: At this stage, self.source_dir is a top-level directory.
        # TFrecords are assumed to live in tfrecords-subdirectory,
        # input files are assumed to live in pickle-subdirectory
        self.destination_dir = os.path.join(source_dir_base, "tfrecords")
        self.source_dir = os.path.join(self.source_dir, "pickle")

        # check if expected data is available in source_dir (depending on dataset)
        # The following files are expected:
        # * ERA5: pickle-files
        # * moving_MNIST: single npy-file
        if self.dataset == "era5":
            # pickle files are expected to be stored in yearly-subdirectories, i.e. we need a wildcard here
            if not any(glob.glob(os.path.join(self.source_dir, "*", "*X*.pkl"))):
                raise FileNotFoundError("Could not find any pickle-files under '{0}'".format(self.source_dir) +
                                        "which are expected for the ERA5-dataset.".format(self.source_dir))
        elif self.dataset == "moving_mnist":
            if not os.path.isfile(os.path.join(self.source_dir, "mnist_test_seq.npy")):
                raise FileNotFoundError("Could not find expected file 'mnist_test_seq.npy' under {0}"
                                        .format(self.source_dir))

        # final keyboard interaction when ERA5-dataset is used
        if self.dataset == "era5":
            # get desired sequence length
            seql_req_str = "Enter desired total sequence length (i.e. number of frames/images):\n"
            seql_err = ValueError("sequence length must be an integer and larger than 2.")

            seql_str = Config_Preprocess2.keyboard_interaction(seql_req_str, Config_Preprocess2.get_seq_length,
                                                               seql_err)
            self.sequence_length = int(seql_str)

        # list of variables to be written to runscript
        if self.dataset == "era5": self.list_batch_vars.append("sequence_length")
    #
    # -----------------------------------------------------------------------------------
    #
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
                for dataset_avail in Config_Preprocess2.allowed_datasets: print("* " + dataset_avail)
            return False
        else:
            return True
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_data_indir(indir, silent=False):
        """
        Rough check of passed directory (if it exist at all)
        :param indir: path to passed input directory
        :param silent: flag if print-statement are executed
        """
        status = True
        if not os.path.isdir(indir):
            status = False
            if not silent: print("Could not find data directory '{0}'.".format(indir))

        return status
    #
    # -----------------------------------------------------------------------------------
    #
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
#
# -----------------------------------------------------------------------------------
#
