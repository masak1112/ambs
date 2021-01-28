"""
Child class used for configuring the runscript of preprocessing step 1 of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-28"

# import modules
import os, glob
import numpy as np
from config_utils import Config_runscript_base    # import parent class

class Config_Preprocess1(Config_runscript_base):

    cls_name = "Config_Preprocess1"#.__name__

    known_vars = ["t2", "msl", "gph500"]         # list of known variables (lower case only!) from ERA5
                                                 # which can be used for training
    nvars = 3                                    # number of variables required for training

    def __init__(self, wrk_flw_step, runscript_base):
        super().__init__(wrk_flw_step, runscript_base)

        self.runscript_template = self.rscrpt_tmpl_prefix + "era5" + self.suffix_template
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