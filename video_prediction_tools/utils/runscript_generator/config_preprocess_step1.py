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
                      "i.e. execute 'source modules_preprocess+extract.sh' from env_setup-directory in terminal first.")

from netcdf_datahandling import NetcdfUtils
from general_utils import check_str_in_list
from runscript_generator.config_utils import Config_runscript_base    # import parent class

class Config_Preprocess1(Config_runscript_base):

    cls_name = "Config_Preprocess1"#.__name__

    nvars_default = 3                                    # number of variables required for training

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
        self.variables = []
        self.sw_corner = [-999., -999.]  # [np.nan, np.nan]
        self.nyx = [-999., -999.]  # [np.nan, np.nan]
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "destination_dir", "years", "variables",
                                "sw_corner", "nyx"]
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

        src_dir_req_str = "Enter path to directory where netCDF-files of the ERA5 dataset are located " + \
                          "(in yearly directories.). Just press enter if the default should be used."
        sorurce_dir_err = NotADirectoryError("Passed directory does not exist.")
        source_dir_str = Config_Preprocess1.keyboard_interaction(src_dir_req_str, Config_Preprocess1.src_dir_check,
                                                                 sorurce_dir_err, ntries=3)
        if not source_dir_str:
            # standard source_dir
            self.source_dir = Config_Preprocess1.handle_source_dir(self, "extractedData")
            print("%{0}: The following standard base-directory obtained from runscript template was set: '{1}'".format(method_name, self.source_dir))
        else:
            self.source_dir = source_dir_str
            Config_Preprocess1.get_subdir_list(self.source_dir)

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
                raise FileNotFoundError("%{0}: Cannot retrieve ERA5 netCDF-files from {1}".format(method_name,
                                                                                                  year_path))

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
        vars_list = [var.strip().lower() for var in vars_list]
        if len(vars_list) == 1:
            self.variables = vars_list * Config_Preprocess1.nvars_default
        else:
            self.variables = [var.strip() for var in vars_list]

        Config_Preprocess1.check_vars_allyears(self)

        # get start and end indices in latitude direction
        swc_req_str = "Enter comma separated pair of latitude [-90..90] and longitude [0..360] defining the southern" +\
                      " and western edge of the target domain (e.g. '50.91, 6.36' which corresponds to 50.91°N, 6.36°E)"
        swc_err = ValueError("Inproper coordinate pair entered. Consider value range of latitude and longitude!")

        swc_str = Config_Preprocess1.keyboard_interaction(swc_req_str, Config_Preprocess1.check_swc,
                                                      swc_err, ntries=2)

        swc_list = swc_str.split(",")
        self.sw_corner = [geo_coord.strip() for geo_coord in swc_list]

        # get size of target domain in number of grid points
        nyx_req_str = "Enter comma-separated number of grid points of target doamin in meridional and zonal direction"
        nyx_err = ValueError("Invalid number of gridpoints chosen")

        nyx_str = Config_Preprocess1.keyboard_interaction(nyx_req_str, Config_Preprocess1.check_nyx,
                                                          nyx_err, ntries=2)

        nyx_list = nyx_str.split(",")
        self.nyx = [nyx_val.strip() for nyx_val in nyx_list]

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

        data = NetcdfUtils(fname)
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
            data = NetcdfUtils(fname)

            stat = check_str_in_list(data.varlist, self.variables, labort=False)

            if not stat:
                raise ValueError("%{0}: Could not find all required variables in data for year {1}".format(method,
                                                                                                           str(year)))

    # auxiliary functions for keyboard interaction
    @staticmethod
    def src_dir_check(srcdir, silent=False):
        """
        Checks if source directory exists. Also allows for empty strings. In this case, a default of the source
        directory must be applied.
        :param srcdir: directory path under which ERA5 netCDF-data is stored
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        method = Config_Preprocess1.src_dir_check.__name__

        status = False
        if srcdir:
            if os.path.isdir(srcdir):
                status = True
            else:
                if not silent:
                    print("%{0}: '{1}' does not exist.".format(method, srcdir))
        else:
            status = True

        return status

    @staticmethod
    def check_data_indir(indir, silent=False, recursive=True):
        """
        Check recursively for existence era5 netCDF-files in indir.
        :param indir: path to passed input directory
        :param silent: flag if print-statement are executed
        :param recursive: flag if recursive search should be performed
        :return: status with True confirming success
        """
        method = Config_Preprocess1.check_data_indir.__name__

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
                if not silent: print("%{0}: {1} does not contain any ERA5 netCDF-files.".format(method, indir))
        else:
            if not silent: print("%{0}: Could not find data directory '{0}'.".format(method, indir))

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
        method = Config_Preprocess1.check_years.__name__

        years_list = years_str.split(",")
        # check if all elements of list are numeric elements
        check_years = [year.strip().isnumeric() for year in years_list]
        status = all(check_years)
        if not status:
            inds_bad = [i for i, e in enumerate(check_years) if not e] #np.where(~np.array(check_years))[0]
            if not silent:
                print("%{0}: The following comma-separated elements could not be interpreted as valid years:"
                      .format(method))
                for ind in inds_bad:
                    print(years_list[ind])
            return status

        # check if all year-values are ok
        check_years_val = [int(year) > 1970 for year in years_list]
        status = all(check_years_val)
        if not status:
            if not silent: print("%{0}: All years must be after year 1970.".format(method))

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
        method = Config_Preprocess1.check_variables.__name__

        vars_list = vars_str.split(",")
        check_vars = [var.strip().lower() in known_vars for var in vars_list]
        status = all(check_vars)
        if not status:
            inds_bad = [i for i, e in enumerate(check_vars) if not e]  # np.where(~np.array(check_vars))[0]
            if not silent:
                print("%{0}: The following comma-separated elements are unknown variables:".format(method))
                for ind in inds_bad:
                    print("* {0}".format(vars_list[ind]))
            return status

        if not len(check_vars) >= 1:
            if not silent: print("%{0}: Pass at least one input variable".format(method))
            status = False

        return status
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_swc(swc_str, silent=False):
        """
        Check if comma-separated input string constitutes a valid coordinate pair for the ERA5 dataset
        :param swc_str: Comma-separated string of coordinates (lat,lon)
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        method = Config_Preprocess1.check_swc.__name__

        status = False
        swc_list = swc_str.split(",")
        if not len(swc_list) == 2:
            if not silent: print("%{0}: Invalid coordinate pair was passed.".format(method))
            return status

        swc_list = [geo_coord.strip() for geo_coord in swc_list]
        if Config_Preprocess1.isfloat(swc_list[0]) and Config_Preprocess1.isfloat(swc_list[1]):
            s_brd, w_brd = float(swc_list[0]), float(swc_list[1])
            check_swc = [-90. <= s_brd <= 90., 0. <= w_brd <= 359.7]
            if all(check_swc):
                status = True
            else:
                if not silent:
                    if not check_swc[0]:
                        print("%{0}: Latitude coordinate must be within -90. and 90.".format(method))
                    if not check_swc[1]:
                        print("%{0}: Longitude coordinate must be within 0. and 360.".format(method))
        else:
            if not silent: print("%{0}: Coordinates must be numbers.".format(method))

        return status

    @staticmethod
    def check_nyx(nyx_str, silent=False):
        """
        Check if comma-separated input string is a valid number of grid points for size of target domain
        :param nyx_str: Comma-separated string of number of gridpoints in meridional and zonal direction
                        of target domain (ny, nx)
        :param silent: flag if print-statement are executed
        :return: status with True confirming success
        """
        method = Config_Preprocess1.check_nyx.__name__
 
        status = False
        nyx_list = nyx_str.split(",")
        if not len(nyx_list) == 2:
            if not silent: print("%{0}: Invalid number pair was passed.".format(method))
            return status

        nyx_list = [nyx_val.strip() for nyx_val in nyx_list]
        if nyx_list[0].isnumeric() and nyx_list[1].isnumeric():
            ny, nx = int(nyx_list[0]), float(nyx_list[1])
            # Note: The value of 0.3 corresponds to the grid spacing of the used ERA5-data set
            ny_max, nx_max = int(180/0.3 - 1), int(360/0.3 - 1)
            check_nyx = [0 <= ny <= ny_max, 0 <= nx <= nx_max]
            if all(check_nyx):
                status = True
            else:
                if not silent:
                    if not check_nyx[0]:
                        print("%{0}: Number of grid points in meridional direction must be smaller than {1:d}"
                              .format(method, ny_max))
                    if not check_nyx[1]:
                        print("%{0}: Number of grid points in zonal direction must be smaller than {1:d}"
                              .format(method, nx_max))
        else:
            if not silent: print("%{0}: Number of grid points must be integers.".format(method))

        return status

    @staticmethod
    def isfloat(str_in):
        """
        Checks if given string can be converted to float.
        :param str_in: input string to be tested
        :return: True if string can be converted to float, False else
        """
        try:
            float(str_in)
            return True
        except ValueError:
            return False

#
# -----------------------------------------------------------------------------------
#
