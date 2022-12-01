# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Child class used for configuring the data extraction runscript of the workflow.
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-27"

# import modules
import os, glob
import subprocess as sp
import time
from pathlib import Path

from runscript_generator.config_utils import Config_runscript_base    # import parent class
from dataset_utils import DATASETS, get_dataset_info

class Config_Extraction(Config_runscript_base):

    cls_name = "Config_Extraction"#.__name__
    era5dir_just = "/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib"

    def __init__(self, venv_name, lhpc):
        super().__init__(venv_name, lhpc)

        # initialize attributes related to runscript name
        self.long_name_wrk_step = "Data Extraction"
        self.rscrpt_tmpl_prefix = "data_extraction_"
        self.dataset = "era5"
        self.runscript_template = self.rscrpt_tmpl_prefix + self.dataset + self.suffix_template
        self.runscript_target = self.rscrpt_tmpl_prefix + self.dataset + ".sh"
        # initialize additional runscript-specific attributes to be set via keyboard interaction
        self.years = None
        self.varmap_file = None
        self.destination_dir = None
        # list of variables to be written to runscript
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "varmap_file", "years", "destination_dir"]
    #
    # -----------------------------------------------------------------------------------
    #
    def run(self):
        """
        Runs the keyboard interaction for data extraction step
        :return: all attributes of class Data_Extraction are set
        """
        
        dataset = Config_Extraction.keyboard_interaction(
            "Choose dataset:" + "".join([f"\n* {name}" for name in DATASETS]) + "\n",
            lambda name, silent=False: name in (*DATASETS, "era5"),
            ValueError("Cannot find dataset for given name"),
            ntries=3
        )
        
        if dataset == "weatherbench": # TODO: change once more datasets are added
            self.extract(dataset)
        else:
            self.extract_old()
            

    def extract(self, dataset):
        """
        Special method for handling of weatherbench data.
        
        (in order to not break previous code)
        """
        available_years = get_dataset_info("weatherbench")["years"]
        
        # override in order to accomdate changes in the structure quick/dirty
        self.runscript_template = self.rscrpt_tmpl_prefix + dataset + self.suffix_template
        self.runscript_target = self.rscrpt_tmpl_prefix + dataset + ".sh"
        self.list_batch_vars = ["VIRT_ENV_NAME", "source_dir", "years", "destination_dir"]
        
        def check_input_dir(path, silent=False):
            try:
                return Path(path).is_dir() 
            except PermissionError as e:
                print(f"Could not access {path} because of missing permissions.")
                return False
            return False
        
        def check_years(years, silent=False):
            years = [year.strip() for year in years.split(",")]
            notyears = list(filter(lambda year: not str.isnumeric(year), years))
            if len(notyears) > 0:
                print("The following elements could not be interpreted as valid years:")
                for elem in notyears:
                    print(elem, end=", ")
                return False
            
            return all([int(year) > 1970 and int(year) in available_years for year in years])
        
        # get input folder
        source_dir = Config_Extraction.keyboard_interaction(
            "Enter path to weatherbench data root folder",
            check_input_dir,
            ValueError("Could not access directory under given path"),
            ntries=3
        )
        
        # get output folder
        destination_dir = Config_Extraction.keyboard_interaction(
            "Enter path for destination directory",
            check_input_dir,
            ValueError("Could not access directory under given path"),
            ntries=3
        )
        
        # get years
        years = Config_Extraction.keyboard_interaction(
            "Enter a comma-separated sequence of years for which data extraction should be performed:",
            check_years,
            ValueError("Cannot get years for preprocessing."),
            ntries=3
        )
        years = [year.strip() for year in years.split(",")]
        
        # set parameters to be written to file
        self.years = years
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        
    
    def extract_old(self):
        method_name = Config_Extraction.extract.__name__
        
        dataset_req_str = "Enter the path where the original ERA5 grib-files are located (standard on JUST: '{0}'):"\
                              .format(self.era5dir_just)
        dataset_err = FileNotFoundError("Cannot retrieve input data from passed path.")

        self.source_dir = Config_Extraction.keyboard_interaction(dataset_req_str, Config_Extraction.check_data_indir,
                                                                 dataset_err, ntries=3) # maybe use locale variable ?

        # get years for preprcessing step 1
        years_req_str = "Enter a comma-separated sequence of years for which data extraction should be performed:"
        years_err = ValueError("Cannot get years for preprocessing.")
        years_str = Config_Extraction.keyboard_interaction(years_req_str, Config_Extraction.check_years,
                                                            years_err, ntries=2)

        self.years = [year.strip() for year in years_str.split(",")]
        # final check data availability for each year
        for year in self.years:
            year_path = os.path.join(self.source_dir, year)
            status = Config_Extraction.check_data_indir(year_path, silent=True, recursive=False, depth=1)
            if status:
                print("%{0}: Data availability checked for year {1}".format(method_name, year))
            else:
                raise FileNotFoundError("%{0}: Cannot retrieve ERA5 grib-files from {1}".format(method_name,
                                                                                                year_path))

        # Create and configure json-file for selecting and mapping variables from ERA5-dataset
        base_path_map = os.path.join("..", "data_preprocess")
        source_varmap = os.path.join(base_path_map, "era5_varmapping_template.json")
        dest_varmap = os.path.join(base_path_map, "era5_varmapping.json")
        # sanity check (default data_split json-file exists)
        if not os.path.isfile(source_varmap):
            raise FileNotFoundError("%{0}: Could not find template variable mapping json-file '{1}'"
                                    .format(method_name, source_varmap))
        # ...copy over json-file for data splitting...
        os.system("cp "+source_varmap+" "+dest_varmap)
        # ...and open vim after some delay
        print("*** Please configure the mapping for ERA5 variables: ***")
        time.sleep(3)
        cmd_vim = os.environ.get('EDITOR', 'vi') + ' ' + os.path.join(dest_varmap)
        sp.call(cmd_vim, shell=True)
        sp.call("sed -i '/^#/d' {0}".format(dest_varmap), shell=True)
        # set varmap_file attribute accordingly
        self.varmap_file = dest_varmap

        # set destination directory based on base directory which can be retrieved from the template runscript
        base_dir = Config_Extraction.get_var_from_runscript(os.path.join(self.runscript_dir, self.runscript_template),
                                                            "destination_dir")
        self.destination_dir = os.path.join(base_dir, "extractedData")
    
    #
    # -----------------------------------------------------------------------------------
    #
    # auxiliary functions for keyboard interaction
    @staticmethod
    def check_data_indir(indir, silent=False, recursive=True, depth=1):
        """
        Check (recursively) for existence of grib-files in indir.
        This is just a simplified check, i.e. the script will fail if the directory tree is not
        built up like '<indir>/YYYY/MM/'.
        :param indir: path to passed input directory
        :param silent: flag if print-statement are executed
        :param recursive: flag if one-level (!) recursive search should be performed
        :param depth: number of levels to descend to perform searching (only 0 or 1 are accepted)
        :return: status with True confirming success
        """

        search_str = "*.grb"
        status = False
        if os.path.isdir(indir):
            # the built-in 'any'-function has a short-sircuit mechanism, i.e. returns True
            # if the first True element is met
            if recursive:
                fexist = any(glob.iglob(os.path.join(indir, "**", search_str), recursive=True))
            else:
                if depth == 1:
                    fexist = any(glob.iglob(os.path.join(indir, "*", search_str)))
                elif depth == 0:
                    fexist = any(glob.iglob(os.path.join(indir, search_str)))
                else:
                    if not silent: print("Invalid depth-argument ({0}) passed, default to depth=1".format(depth))
                    fexist = any(glob.iglob(os.path.join(indir, "*", search_str)))

            if fexist:
                status = True
            else:
                if not silent: print("{0} does not contain any files matching search string '{1}'.".format(indir,
                                                                                                           search_str))
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
