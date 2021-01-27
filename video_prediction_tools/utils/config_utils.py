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
        """
        Acts as generic wrapper: Checks if run_config is already set up as a callable
        :return: Executes run_config
        """

        method_name = "run" + " of Class " + Config_runscript_base.cls_name
        if self.run_config is None:
            raise ValueError("%{0}: run-method is still uninitialized.".format(method_name))

        if not callable(self.run_config):
            raise ValueError("%{0}: run-method is not callable".format(method_name))

        # simply execute it
        self.run_config(self)

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
            self.rscrpt_tmpl_suffix = "data_extraction"
        elif wrk_flw_step == "preprocess1":
            self.long_name_wrk_step = "Preprocessing step 1"
            self.rscrpt_tmpl_suffix = "preprocess_data"
        elif wrk_flw_step == "preprocess2":
            self.long_name_wrk_step = "Preproccessing step 2"
            self.rscrpt_tmpl_suffix = "preprocess_data"
        elif wrk_flw_step == "train":
            self.long_name_wrk_step = "Training"
            self.rscrpt_tmpl_suffix = "train_model"
        elif wrk_flw_step == "postprocess":
            self.long_name_wrk_step = "Postprocessing"
            self.rscrpt_tmpl_suffix = "visualize_postprocess"
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

        self.runscript_template = self.rscrpt_tmpl_suffix + "era5" + self.suffix_template
        self.year = None
        self.run_config = Config_Extraction.run_extraction

    def run_extraction(self):
        """
        Runs the keyboard interaction for data extraction step
        :return: all attributes of class Data_Extraction are set
        """

        dataset_req_str = "Enter the path where the original ERA5 netCDF-files are located:\n"
        dataset_err = FileNotFoundError("Cannot retrieve input data from passed path.")

        self.source_dir = Config_Extraction.keyboard_interaction(dataset_req_str, Config_Extraction.check_data_indir,
                                                                 dataset_err, ntries=3)

        year_req_str = "Enter the year for which data extraction should be performed:\n"
        year_err = ValueError("Please type in a year (after 1970) in YYYY-format.")

        self.year = Config_Extraction.keyboard_interaction(year_req_str, Config_Extraction.check_year,
                                                           year_err, ntries = 2, test_arg="2012")

    # auxiliary functions for keyboard interaction
    @staticmethod
    def check_data_indir(indir, silent=False):
        """
        Check recursively for existence era5 netCDF-files in indir.
        This is just a simplified check, i.e. the script will fail if the directory tree is not
        built up like '<indir>/YYYY/MM/'.
        Also used in Config_preprocess1!
        :param indir: path to passed input directory
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        status = False
        if os.path.isdir(indir):
            # the built-in 'any'-function has a short-sircuit mechanism, i.e. returns True
            # if the first True element is met
            fexist = any(glob.glob(os.path.join(indir, "**", "*era5*.nc"), recursive=True))
            if fexist:
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

        if not int(year_in) > 1970:
            status = False
            if not silent: print("{0} must match the format YYYY and must be larger than 1970.")

        return status

class Config_Preprocess1(Config_runscript_base):

    cls_name = "Config_Preprocess1"#.__name__

    known_vars = ["t2", "msl", "gph500"]         # list of known variables (lower case only!) from ERA5
                                                 # which can be used for training
    nvars = 3                                    # number of variables required for training

    def __init__(self, wrk_flw_step, runscript_base):
        super().__init__(wrk_flw_step, runscript_base)

        self.runscript_template = self.rscrpt_tmpl_suffix + "era5" + self.suffix_template
        self.years = None
        self.variables = [None] * self.nvars
        self.lat_inds = [np.nan, np.nan]
        self.lon_inds = [np.nan, np.nan]

        self.run_config = Config_Preprocess1.run_preprocess1

    def run_preprocess1(self):
        """
        Runs the keyboard interaction for Preprocessing step 1
        :return: all attributes of class Config_Preprocess1 are set
        """
        # get source_dir
        dataset_req_str = "Enter the path where the extracted ERA5 netCDF-files are located:\n"
        dataset_err = FileNotFoundError("Cannot retrieve extracted ERA5 netCDF-files from passed path.")

        self.source_dir = Config_Preprocess1.keyboard_interaction(dataset_req_str, Config_Preprocess1.check_data_indir,
                                                                  dataset_err, ntries=3)

        # get years for preprcessing step 1
        years_req_str = "Enter a comma-separated sequence of years (format: YYYY):\n"
        years_err = ValueError("Cannot get years for preprocessing.")
        years_str = Config_Extraction.keyboard_interaction(years_req_str, Config_Preprocess1.check_years,
                                                           years_err, ntries=2)

        self.years = [int(year.strip()) for year in years_str.split(",")]

        # get variables for later training
        print("**** Info ****\n List of known variables which can be processed")
        for known_var in Config_Preprocess1.known_vars:
            print("* {0}".format(known_var))

        vars_req_str = "Enter either three or one variable names to be processed:\n"
        vars_err = ValueError("Cannot get variabes for preprocessing.")
        vars_str = Config_Preprocess1.keyboard_interaction(vars_req_str, Config_Preprocess1.check_variables,
                                                           vars_err, ntries=2)

        vars_list = vars_str.split(",")
        if len(vars_list) == 1:
            self.variables = vars_list * Config_Preprocess1.nvars
        else:
            self.variables = vars_list

        # get start and end indices in latitude direction
        lat_req_str = "Enter comma-separated indices of start and end index in latitude direction for target domain:\n"
        lat_err = ValueError("Cannot retrieve proper pair of latitude indices.")

        lat_inds_str = Config_Preprocess1.keyboard_interaction(lat_req_str, Config_Preprocess1.check_latlon_inds,
                                                               lat_err, ntries=2)

        lat_inds_list = lat_inds_str.split(",")
        self.lat_inds = [int(ind.strip()) for ind in lat_inds_list]

        # get start and end indices in longitude direction
        lon_req_str = lat_req_str.replace("latitude", "longitude")
        lon_err = ValueError("Cannot retrieve proper pair of longitude indices.")

        lon_inds_str = Config_Preprocess1.keyboard_interaction(lon_req_str, Config_Preprocess1.check_latlon_inds,
                                                               lon_err, ntries=2)

        lon_inds_list = lon_inds_str.split(",")
        self.lon_inds = [int(ind.strip()) for ind in lon_inds_list]

    # auxiliary functions for keyboard interaction
    @staticmethod
    def check_data_indir(indir, silent=False):
        """
        Check recursively for existence era5 netCDF-files in indir.
        This is just a simplified check, i.e. the script will fail if the directory tree is not
        built up like '<indir>/YYYY/MM/'.
        Also used in Config_preprocess1!
        :param indir: path to passed input directory
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        status = False
        if os.path.isdir(indir):
            # the built-in 'any'-function has a short-sircuit mechanism, i.e. returns True
            # if the first True element is met
            fexist = any(glob.glob(os.path.join(indir, "**", "*era5*.nc"), recursive=True))
            if fexist:
                status = True
            else:
                if not silent: print("{0} does not contain any ERA5 netCDF-files.".format(indir))
        else:
            if not silent: print("Could not find data directory '{0}'.".format(indir))

        return status

    @staticmethod
    def check_years(years_str, silent=False):
        """
        Check if comma-separated string of years contains vaild entities (i.e. numbers and years after 1950)
        :param years: Comma separated string of years with format YYYY
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """

        years_list = years_str.split(",")
        # check if all elements of list are numeric elements
        check_years = [year.strip().isnumeric() for year in years_list]
        status = all(check_years)
        if not status:
            inds_bad = np.where(~np.array(check_years))[0]
            if not silent:
                print("The following comma-separated elements could not be interpreted as valid years:")
                for ind in inds_bad:
                    print(years_list[ind])
            return status

        # check if all year-values are ok
        check_years_val = [int(year) > 1950 for year in years_list]
        status = all(check_years_val)
        if not status:
            if not silent: print("All years must be after year 1950.")

        return status

    @staticmethod
    def check_variables(vars_str, silent=False):
        """
        Checks if comma-separated string of variables is valid (i.e. known variables and one or three variables)
        :param vars_str: Comma-separated string of variables
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        vars_list = vars_str.split(",")
        check_vars = [var.strip().lower() in Config_Preprocess1.known_vars for var in vars_list]
        status = all(check_vars)
        if not status:
            inds_bad = np.where(~np.array(check_vars))[0]
            if not silent:
                print("The following comma-separated elements are unknown variables:")
                for ind in inds_bad:
                    print(vars_list[ind])
            return status

        if not (len(check_vars) == Config_Preprocess1.nvars or len(check_vars) == 1):
            status = False

        return status

    @staticmethod
    def check_latlon_inds(inds_str, silent=False):
        """
        Check if comma-separated string of indices is non-negative and passed in increasing order.
        Exactly, two indices must be passed!
        :param inds_str: Comma-separated string of indices
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """

        status = False
        inds_list = inds_str.split(",")
        if not len(inds_list) == 2:
            if not silent: print("Invalid number of indices identified.")
            return status

        inds_list = [ind.strip() for ind in inds_list]
        if inds_list[0].isnumeric() and inds_list[1].isnumeric():
            ind1, ind2 = int(inds_list[0]), int(inds_list[1])
            if ind1 >= ind2 or ind1 < 0 or ind2 <= 0:
                if not silent: print("Indices must be non-negative and passed in increasing order.")
            else:
                status = True
        else:
            if not silent: print("Indices must be numbers.")

        return status











