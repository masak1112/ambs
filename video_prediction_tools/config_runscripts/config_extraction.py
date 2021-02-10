"""
Child class used for configuring the data extraction runscript of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-27"

# import modules
import os, glob
from config_utils import Config_runscript_base    # import parent class

class Config_Extraction(Config_runscript_base):

    cls_name = "Config_Extraction"#.__name__

    def __init__(self, venv_name, lhpc):
        super().__init__(venv_name, lhpc)

        # initialize attributes related to runscript name
        self.long_name_wrk_step = "Data Extraction"
        self.rscrpt_tmpl_prefix = "data_extraction_"
        self.dataset = "era5"
        self.runscript_template = self.rscrpt_tmpl_prefix + self.dataset + self.suffix_template
        self.runscript_target = self.rscrpt_tmpl_prefix + self.dataset + ".sh"
        # initialize additional runscript-specific attributes to be set via keyboard interaction
        self.year = None
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "destination_dir"]
        # copy over method for keyboard interaction
        self.run_config = Config_Extraction.run_extraction
    #
    # -----------------------------------------------------------------------------------
    #
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

        # final check for input data
        path_year = os.path.join(self.source_dir, self.year)
        if not Config_Extraction.check_data_indir(path_year, silent=True, recursive=False):
            raise FileNotFoundError("Cannot retrieve input data from {0}".format(path_year))

        # set destination directory based on base directory which can be retrieved from the template runscript
        base_dir = Config_Extraction.get_var_from_runscript(os.path.join(self.runscript_dir, self.runscript_template),
                                                            "destination_dir")
        self.destination_dir = os.path.join(base_dir, "extracted_data", self.year)

    #
    # -----------------------------------------------------------------------------------
    #
    # auxiliary functions for keyboard interaction
    @staticmethod
    def check_data_indir(indir, silent=False, recursive=True):
        """
        Check recursively for existence era5 netCDF-files in indir.
        This is just a simplified check, i.e. the script will fail if the directory tree is not
        built up like '<indir>/YYYY/MM/'.
        :param indir: path to passed input directory
        :param silent: flag if print-statement are executed
        :param recursive: flag if one-level (!) recursive search should be performed
        :return: status with True confirming success
        """
        status = False
        if os.path.isdir(indir):
            # the built-in 'any'-function has a short-sircuit mechanism, i.e. returns True
            # if the first True element is met
            if recursive:
                fexist = any(glob.glob(os.path.join(indir, "**", "*era5*.nc"), recursive=True))
            else:
                fexist = any(glob.glob(os.path.join(indir, "*", "*era5*.nc")))

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
    def check_year(year_in, silent=False):
        status = True
        if not year_in.isnumeric():
            status = False
            if not silent: print("{0} is not a numeric number.".format(year_in))

        if not int(year_in) > 1970:
            status = False
            if not silent: print("{0} must match the format YYYY and must be larger than 1970.")

        return status
#
# -----------------------------------------------------------------------------------
#
