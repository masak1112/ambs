"""
Class and subclasses to control behaviour of config_runscript.py which generates executable runscripts from templates
"""
__author__ = "Michael Langguth"
date = "2021-01-25"

# import modules
import os, glob
import numpy as np

class Config_runscript_base:

    cls_name = "Config_runscript_base"

    def __init__(self, wrk_flw_step, runscript_base):
        self.runscript_base     = runscript_base
        self.long_name_wrk_step = None
        self.runscript_template = None
        self.suffix_template = "_template.sh"
        Config_runscript_base.check_and_set_basic(self, wrk_flw_step)

        self.source_dir = None
        self.run_config = None

    def run(self):

        method_name = Config_runscript_base.__run__.__name__ + " of Class " + Config_runscript_base.cls_name
        if self.run_config is None:
            raise ValueError("%{0}: run-method is still uninitialized.".format(method_name))

        if not callable(self.run_config):
            raise ValueError("%{0}: run-method is not callable".format(method_name))

        # simply execute it
        self.run_config()

    def check_and_set_basic(self, wrk_flw_step):

        method_name = Config_runscript_base.check_and_set_basic.__name__ + " of Class " + Config_runscript_base.cls_name

        if not isinstance(wrk_flw_step, str):
            raise ValueError("%{0}: wrk_flw_step-arument must be string indicating the name of workflow substep."
                             .format(method_name))

        if not os.path.isdir(self.runscript_base):
            raise NotADirectoryError("%{0}: Could not find directory where runscript templates are expected: {1}"
                                     .format(method_name, self.runscript_base))

        if wrk_flw_step == "extract":
            self.long_name_wrk_step = "Data Extraction"
            self.rscrpt_tmpl_suffix = "data_extraction_era5"
        elif wrk_flw_step == "preprocess1":
            self.long_name_wrk_step = "Preprocessing step 1"
            self.runscript_template = "preprocess_data"
        elif wrk_flw_step == "preprocess2":
            self.long_name_wrk_step = "Preproccessing step 2"
            self.runscript_template = "preprocess_data"
        elif wrk_flw_step == "train":
            self.long_name_wrk_step = "Training"
            self.runscript_template = "train_model"
        elif wrk_flw_step == "postprocess":
            self.long_name_wrk_step = "Postprocessing"
            self.runscript_template = "visualize_postprocess"
        else:
            raise ValueError("%{0}: Workflow step {1} is unknown / not implemented.".format(method_name, wrk_flw_step))

    @staticmethod
    def keyboard_interaction(console_str, check_input, err, ntries=1, test_arg=None):
        """
        Function to check if the user has passed a proper input via keyboard interaction
        :param console_str: Request printed to the console
        :param check_input: function returning boolean which needs to be passed by input from keyboard interaction.
                            Must have two arguments with the latter being an optional bool called silent.
        :param ntries: maximum number of tries (default: 1)
        :return: The approved input from keyboard interaction
        """
        if test_arg is None: test_arg = "xxx"
        # sanity checks
        if not callable(check_input):
            raise ValueError("check_input must be a function!")
        else:
            try:
                if not type(check_input(test_arg, silent=True)) is bool:
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


class Config_Extraction(Config_runscript_base):

    cls_name = "Config_Extraction"#.__name__

    def __init__(self, wrk_flw_step, runscript_base):
        super().__init__(wrk_flw_step, runscript_base)

        self.year = None
        #self.run_config = Config_Extraction.run_extraction(self)

    def run_extraction(self):

        dataset_req_str = "Enter the path where the original ERA5 netCDF-files are located:\n"
        dataset_err = ValueError("Cannot retrieve input data from passed path.")

        self.source_dir = Config_Extraction.keyboard_interaction(dataset_req_str, Config_Extraction.check_data_indir,
                                                                 dataset_err, ntries=3)

        year_req_str = "Enter the year for which data extraction should be performed:\n"
        year_err = ValueError("Please type in a year (after 1970) in YYYY-format.")

        self.year = Config_Extraction.keyboard_interaction(year_req_str, Config_Extraction.check_year,
                                                           year_err, ntries = 2, test_arg="2012")

    @staticmethod
    def check_data_indir(indir, silent=False):
        # NOTE: Generic template for training still has to be integrated!
        #       After this is done, the latter part of the if-clause can be removed
        #       and further adaptions for the target_dir and for retrieving base_dir (see below) are required

        status = False
        if os.path.isdir(indir):
            file_list = glob.glob(os.path.join(indir, "era5*.nc"))
            if len(file_list) > 0:
                status = True
            else:
                if not silent: print("{0} does not contain any ERA5 netCDF-files.".format(indir))
        else:
            if not silent: print("Could not find data directory '{0}'.".format(indir))

        return status

    @staticmethod
    def check_year(year_in, silent=False):
        status = True
        if not year_in.isnumeric():
            status = False
            if not silent: print("{0} is not a numeric number.".format(year_in))

        if not int(year_in) > 1950:
            status = False
            if not silent: print("{0} must match the format YYYY and must be larger than 1950.")

        return status

class Config_Preprocess1(Config_runscript_base):

    cls_name = "Config_Preprocess1"#.__name__

    def __init__(self, wrk_flw_step, runscript_base):
        super().__init__(wrk_flw_step, runscript_base)

        self.years = None
        self.variables = [None, None, None]         # hard-code to three variables so far
        self.lat_inds = [np.nan, np.nan]
        self.lon_inds = [np.nan, np.nan]

    def run_config(self):

        dataset_req_str = "Enter the path where the extracted ERA5 netCDF-files are located:\n"
        dataset_err = ValueError("Please select a dataset from the ones listed above.")

        self.source_dir = Config_Extraction.keyboard_interaction(dataset_req_str, Config_Preprocess1.check_data_indir,
                                                                 dataset_err, ntries=3)

    @staticmethod
    def check_data_indir(indir, silent=False):
        # NOTE: Generic template for training still has to be integrated!
        #       After this is done, the latter part of the if-clause can be removed
        #       and further adaptions for the target_dir and for retrieving base_dir (see below) are required

        status = False
        if os.path.isdir(indir):
            file_list = glob.glob(os.path.join(indir, "era5*.nc"))
            if len(file_list) > 0:
                status = True
            else:
                if not silent: print("{0} does not contain any ERA5 netCDF-files.".format(indir))
        else:
            if not silent: print("Could not find data directory '{0}'.".format(indir))

        return status