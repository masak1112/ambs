"""
Code for processing staged ERA5 data, this is used for the DataPreprocessing step 1 of the workflow

reviewed by Michael Langguth: 2021-03-21
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Michael Langguth, Bing Gong, Scarlet Stadtler"

import sys, os
import fnmatch
import pickle
import numpy as np
import xarray as xr
import datetime as dt
from metadata import Geo_subdomain
from statistics import Calc_data_stat

class PreprocessNcToPkl(object):

    def __init__(self, src_dir, target_dir, year, job_id, target_dom, variables=("2t", "msl", "t_850")):
        """
        Function to process data from netCDF file to pickle file
        args:
            src_dir    : string, directory based on year  where netCDF-files are stored to be processed
            target_dir : base-directory where data is stored (files are stored under [target_dir]/pickle/[year]/)
            job_id     : job_id with range "01"-"12" (organized by PyStager) job_name also corresponds to the month
            year       : year of data to be processed
            target_dom : class instance of Geo_subdomain which defines target domain
            vars       : variables to be processed
        """
        # directory_to_process is month-based directory
        self.directory_to_process=os.path.join(src_dir,str(year), str(job_id))
        # sanity checks
        if int(job_id) > 12 or int(job_id) < 1 or not isinstance(job_id, str):
            raise ValueError("job_name should be int type between 1 to 12")

        if not os.path.exists(self.directory_to_process):
            raise NotADirectoryError("The directory_to_process '"+self.directory_to_process+"' does not exist")

        if not isinstance(target_dom, Geo_subdomain):
            raise ValueError("target_dom must be a Geo_subdomain-instance.")

        self.target_dir = os.path.join(target_dir, "pickle", str(year))      # preprocessed data to pickle-subdirectory
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)
        self.job_id = job_id
        self.tar_dom = target_dom
        # target file name needs to be saved
        self.target_file = os.path.join(self.target_dir, 'X_' + str(self.job_id) + '.pkl')
        self.vars = variables
        self.nvars = len(variables)
        # attributes to set during call of class instance
        self.imageList = None
        self.stat_obj = None
        self.data = None

    def __call__(self):
        """
        Process the necCDF files in the month_base folder, store the variables of the images into list,
        store temporal information to list and save them to pickle file
        """
        if os.path.exists(self.target_file):
            print(self.target_file, " file exists in the directory ", self.target_dir)
        else:
            print ("==========Processing files in directory {} =============== ".format(self.directory_to_process))
            self.imageList = self.get_images_list()
            self.stat_obj = self.init_stat()
            self.data = self.process_era5_data()
            self.save_data_to_pickle()
            self.save_stat_info()
    # ------------------------------------------------------------------------------------------------------------------
   
    def get_images_list(self, patt="ecmwf_era5_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].nc"):
        """
        Get the images list from the directory_to_process and sort them by date names
        :param patt: The string pattern to filter for (otional)
        :return filelist_filt: filtered list of files whose names match patt
        """
        method = "{0} of class {1}".format(PreprocessNcToPkl.get_images_list.__name__, PreprocessNcToPkl.__name__)

        filelist_all = list(os.walk(self.directory_to_process, topdown = False))[-1][-1]
        filelist_filt = fnmatch.filter(filelist_all, patt)
        filelist_filt = sorted(filelist_filt)
        # sanity check
        if len(filelist_filt) == 0:
            raise FileNotFoundError("%{0}: Could not find ERA5 netCDf-files under '{1}'"
                                    .format(method, self.directory_to_process))

        return filelist_filt

    # ------------------------------------------------------------------------------------------------------------------

    def init_stat(self):
        """
        Initializes the statistics instance
        """
        method = "{0} of class {1}".format(PreprocessNcToPkl.init_stat.__name__, PreprocessNcToPkl.__name__)
        # sanity check
        if self.nvars <= 0:
            raise AttributeError("%{0}: At least one variable must be tracked from the statistic object."
                                 .format(method))

        stat_obj = Calc_data_stat(self.nvars)

        return stat_obj

    # ------------------------------------------------------------------------------------------------------------------

    def process_era5_data(self):
        """
        Get the selected variables from netCDF file, and concanate all the variables from all the images in the
        directiory_to_process into a list EU_stack_list
        EU_stack_list dimension should be [numer_of_images,height, width,number_of_variables] 
        temporal_list is 1-dim list with timestamp data type, contains all the timestamps of netCDF files.
        """
        method = "{0} of class {1}".format(PreprocessNcToPkl.process_era5_data.__name__,
                                           PreprocessNcToPkl.__name__)

        tar_dom = self.tar_dom
        for j, nc_fname in enumerate(self.imageList):
            nc_fname_full = os.path.join(self.directory_to_process, nc_fname)
            try:
                data_curr = tar_dom.get_data_dom(nc_fname_full, self.vars)
                if j == 0:
                    data_all = data_curr.copy(deep=True)
                else:
                    data_all = xr.concat([data_all, data_curr], dim="time")
                # feed statistics-instance (ML, 2021-03-21: This is kind of slow and could be optimized by using
                # the data_all-dataset directly at the end. However, we keep the former approach for now.)
                for i, var in enumerate(self.vars):
                    self.stat_obj.acc_stat_loc(i, np.squeeze(data_curr[var].values))
            except Exception as err:
                print("%{0}: ERROR in job {1}: Could not handle data from netCDf-file '{2}'".format(method, self.job_id,
                                                                                                    nc_fname_full))
                #print("%{0}: The related error is: {1}".format(method, str(err)))
                raise err # would better catched by Pystager

        return data_all

    # ------------------------------------------------------------------------------------------------------------------

    def save_data_to_pickle(self):
        method = "{0} of class {1}".format(PreprocessNcToPkl.save_data_to_pickle.__name__,
                                           PreprocessNcToPkl.__name__)
        # saity check
        if self.data is None:
            raise AttributeError("%{0}: Class instance does not contain any data".format(method))
        
        # construct pickle filenames
        tar_fdata = os.path.join(self.target_dir, "X_{0}.pkl".format(self.job_id))
        tar_ftimes = os.path.join(self.target_dir, "T_{0}.pkl".format(self.job_id))
        
        # write data to pickle-file
        data = self.data
        tar_dom = self.tar_dom
        # roll data if domain crosses zero-meridian (to get spatially coherent data-arrays)
        if tar_dom.lon_slices[0] > tar_dom.lon_slices[1]:
            nroll_lon = tar_dom.nlon - tar_dom.lon_slices[0]
            data = data.roll(lon=nroll_lon, roll_coords=True)

        try:
            data_arr = np.squeeze(data.values)
            with open(tar_fdata, "wb") as pkl_file:
                pickle.dump(data_arr, pkl_file)
        except Exception as err:
            print("%{0}: ERROR in job {1}: could not write data to pickle-file '{2}'".format(method, self.job_id,
                                                                                             tar_fdata))
            # print("%{0}: The related error is: {1}".format(method, str(err)))
            raise err                      # would better catched by Pystager

        # write times to pickle-file incl. conversion to datetime-object
        try:
            time = data.coords["time"]
            time = dt.datetime.strptime(np.datetime_as_string(time, "m")[0], "%Y-%m-%dT%H:%M")
            with open(tar_ftimes, "wb") as tpkl_file:
                pickle.dump(time, tpkl_file)
        except Exception as err:
            print("%{0}: ERROR in job {1}: could not write times to pickle-file '{2}'".format(method, self.job_id,
                                                                                              tar_ftimes))
            # print("%{0}: The related error is: {1}".format(method, str(err)))
            raise err                      # would better catched by Pystager

    # ------------------------------------------------------------------------------------------------------------------

    def save_stat_info(self):
        """
        save the stat information to the target dir
        """
        self.stat_obj.finalize_stat_loc(self.vars)
        self.stat_obj.write_stat_json(self.target_dir, file_id=self.job_id)
       
    # -------------------------------------------- end of class --------------------------------------------------------
        
    

 

                
                 
            
            
        
    
    
    
    
    
    



    

