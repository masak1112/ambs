"""
Child class used for configuring the preprocessing step 1 runscript of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-27"

# import modules
import os, glob
try:
    import xarray as xr
except:
    raise ImportError("Loading preprocssing modules in advance is mandotory, " +
                      "i.e. execute 'source modules_preprocess.sh' in terminal first.")

from metadata import Netcdf_utils
from general_utils import check_str_in_list
from config_utils import Config_runscript_base    # import parent class

class Config_Preprocess1(Config_runscript_base):

    cls_name = "Config_Preprocess1"#.__name__

    nvars = 3                                    # number of variables required for training

    def __init__(self, venv_name, lhpc):
        super().__init__(venv_name, lhpc)

        # initialize attributes related to runscript name
        self.long_name_wrk_step = "Preprocessing step 1"
        self.rscrpt_tmpl_prefix = "preprocess_data_"

        self.dataset = "era5"
        self.runscript_template = self.rscrpt_tmpl_prefix + self.dataset + "_step1" + self.suffix_template
        self.runscript_target = self.rscrpt_tmpl_prefix + self.dataset + "_step1" + ".sh"
        # initialize additional runscript-specific attributes to be set via keyboard interaction
        self.destination_dir = None
        self.years = None
        self.variables = [None] * self.nvars
        self.lat_inds = [-1 -1]  # [np.nan, np.nan]
        self.lon_inds = [-1 -1]  # [np.nan, np.nan]
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "destination_dir", "years", "variables",
                                "lat_inds", "lon_inds"]
        # copy over method for keyboard interaction
        self.run_config = Config_Preprocess1.run_preprocess1
    #
    # -----------------------------------------------------------------------------------
    #
    def run_preprocess1(self):
        """
        Runs the keyboard interaction for Preprocessing step 1
        :return: all attributes of class Config_Preprocess1 are set
        """
        method_name = Config_Preprocess1.run_preprocess1.__name__

        # get source_dir (no user interaction needed when directory tree is fixed)
        self.source_dir = Config_Preprocess1.handle_source_dir(self, "extractedData")

        #dataset_req_str = "Choose a subdirectory listed above where the extracted ERA5 files are located:\n"
        #dataset_err = FileNotFoundError("Cannot retrieve extracted ERA5 netCDF-files from passed path.")

        #self.source_dir = Config_Preprocess1.keyboard_interaction(dataset_req_str, Config_Preprocess1.check_data_indir,
        #                                                          dataset_err, ntries=3, suffix2arg=source_dir_base+"/")

        # get years for preprocessing step 1
        years_req_str = "Enter a comma-separated sequence of years from list above:"
        years_err = ValueError("Cannot get years for preprocessing.")
        years_str = Config_Preprocess1.keyboard_interaction(years_req_str, Config_Preprocess1.check_years,
                                                            years_err, ntries=2)

        self.years = [year.strip() for year in years_str.split(",")]
        # final check data availability for each year
        for year in self.years:
            year_path = os.path.join(self.source_dir, year)
            status = Config_Preprocess1.check_data_indir(year_path, recursive=False)
            if status:
                print("%{0}: Data availability checked for year {1}".format(method_name, year))
            else:
                raise FileNotFoundError("%{0}: Cannot retrieve ERA5 netCDF-files from {1}".format(method_name, year_path))

        # get variables for later training
        # retrieve known variables from first year (other years are checked later)
        print("%{0}: List of known variables which can be processed".format(method_name))
        for known_var in Config_Preprocess1.get_known_vars(self, self.years[0]):
            print("* {0}".format(known_var))

        vars_req_str = "Enter either three or one variable name(s) that should be processed:"
        vars_err = ValueError("Cannot get variabes for preprocessing.")
        vars_str = Config_Preprocess1.keyboard_interaction(vars_req_str, Config_Preprocess1.check_variables,
                                                           vars_err, ntries=2)

        vars_list = vars_str.split(",")
        if len(vars_list) == 1:
            self.variables = vars_list * Config_Preprocess1.nvars
        else:
            self.variables = [var.strip() for var in vars_list]

        check_vars_allyears(self)

        # get start and end indices in latitude direction
        lat_req_str = "Enter comma-separated indices of start and end index in latitude direction for target domain:"
        lat_err = ValueError("Cannot retrieve proper pair of latitude indices.")

        lat_inds_str = Config_Preprocess1.keyboard_interaction(lat_req_str, Config_Preprocess1.check_latlon_inds,
                                                               lat_err, ntries=2)

        lat_inds_list = lat_inds_str.split(",")
        self.lat_inds = [ind.strip() for ind in lat_inds_list]

        # get start and end indices in longitude direction
        lon_req_str = lat_req_str.replace("latitude", "longitude")
        lon_err = ValueError("Cannot retrieve proper pair of longitude indices.")

        lon_inds_str = Config_Preprocess1.keyboard_interaction(lon_req_str, Config_Preprocess1.check_latlon_inds,
                                                               lon_err, ntries=2)

        lon_inds_list = lon_inds_str.split(",")
        self.lon_inds = [ind.strip() for ind in lon_inds_list]

        # set destination directory based on base directory which can be retrieved from the template runscript
        base_dir = Config_Preprocess1.get_var_from_runscript(os.path.join(self.runscript_dir, self.runscript_template),
                                                             "destination_dir")
        self.destination_dir = os.path.join(base_dir, "preprocessedData", "era5-Y{0}-{1}M01to12"
                                            .format(min(self.years), max(self.years)))

    #
    # -----------------------------------------------------------------------------------
    #
    def get_known_vars(self, year, lglobal=False):
        """
        Retrieves known variables from exemplary netCDF-file which will be processed
        :param year: year for which data should be read in to retrieve variable list
        :param lglobal: if True known_vars will become a global variable
        :return known_vars: list of available variable names
        """
        method = Config_Preprocess1.get_known_vars.__name__

        if self.source_dir is None:
            raise AttributeError("%{0}: source_dir property is still None".format(method))

        fname = next(glob.iglob(os.path.join(self.source_dir, str(year), "*", "*era5*.nc")))

        data = Netcdf_utils(fname)
        if lglobal:
            global known_vars

        known_vars = data.varlist

        return known_vars

    def check_vars_allyears(self):

        method = Config_Preprocess1.check_vars_allyears.__name__

        if self.source_dir is None:
            raise AttributeError("%{0}: source_dir property is still None".format(method))

        if self.years is None:
            raise AttributeError("%{0}: years property is still None".format(method))

        if self.variables is None:
            raise AttributeError("%{0}: variables property is still None".format(method))

        for year in self.years:
            fname = next(glob.iglob(os.path.join(self.source_dir, str(year), "*", "*era5*.nc")))
            data = Netcdf_utils(fname)

            stat = check_str_in_list(data.varlist, self.variables, labort=False)

            if not stat:
                raise ValueError("%{0}: Could not find all required variables in data for year {1}".format(method,
                                                                                                           str(year)))

    # auxiliary functions for keyboard interaction
    @staticmethod
    def check_data_indir(indir, silent=False, recursive=True):
        """
        Check recursively for existence era5 netCDF-files in indir.
        :param indir: path to passed input directory
        :param silent: flag if print-statement are executed
        :param recursive: flag if recursive search should be performed
        :return: status with True confirming success
        """
        status = False
        if os.path.isdir(indir):
            # the built-in 'any'-function has a short-circuit mechanism, i.e. returns True
            # if the first True element is met
            if recursive:
                fexist = any(glob.iglob(os.path.join(indir, "**", "*era5*.nc"), recursive=True))
            else:
                fexist = any(glob.iglob(os.path.join(indir, "*", "*era5*.nc")))
            if fexist:
                status = True
            else:
                if not silent: print("{0} does not contain any ERA5 netCDF-files.".format(indir))
        else:
            if not silent: print("Could not find data directory '{0}'.".format(indir))

        return status
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_years(years_str, silent=False):
        """
        Check if comma-separated string of years contains vaild entities (i.e. numbers and years after 1950)
        :param years_str: Comma separated string of years with format YYYY
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """

        years_list = years_str.split(",")
        # check if all elements of list are numeric elements
        check_years = [year.strip().isnumeric() for year in years_list]
        status = all(check_years)
        if not status:
            inds_bad = [i for i, e in enumerate(check_years) if e] #np.where(~np.array(check_years))[0]
            if not silent:
                print("The following comma-separated elements could not be interpreted as valid years:")
                for ind in inds_bad:
                    print(years_list[ind])
            return status

        # check if all year-values are ok
        check_years_val = [int(year) > 1970 for year in years_list]
        status = all(check_years_val)
        if not status:
            if not silent: print("All years must be after year 1970.")

        return status
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_variables(vars_str, silent=False):
        """
        Checks if comma-separated string of variables is valid (i.e. known variables and one or three variables)
        :param vars_str: Comma-separated string of variables
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        vars_list = vars_str.split(",")
        check_vars = [var.strip().lower() in varlist for var in vars_list]
        status = all(check_vars)
        if not status:
            inds_bad = [i for i, e in enumerate(check_vars) if e] # np.where(~np.array(check_vars))[0]
            if not silent:
                print("The following comma-separated elements are unknown variables:")
                for ind in inds_bad:
                    print(vars_list[ind])
            return status

        if not (len(check_vars) == Config_Preprocess1.nvars or len(check_vars) == 1):
            if not silent: print("Unexpected number of variables passed.")
            status = False

        return status
    #
    # -----------------------------------------------------------------------------------
    #
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
    #
    # -----------------------------------------------------------------------------------
    #




#
# -----------------------------------------------------------------------------------
#
