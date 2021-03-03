"""
Functions required for extracting ERA5 data.
"""
import numpy as np
from datetime import datetime
from netCDF4 import Dataset, date2num
from shiftgrid import shiftgrid
import os
import json

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler, Michael Langguth,Yanji"
__date__ = "unknown"
# specify source and target directories


class ERA5DataExtraction(object):

    def __init__(self,job_name,src_dir,target_dir,varslist_json):
        """
        Function to extract ERA5 data from slmet 
        args:
             job_name     :int from 1 to 12 correspoding to month
             scr_dir      :str, upper level of directory at year level
             target_dir   : str, upper level of directory at year level
             varslist_json: str, the path to the varibale list that to be extracted from original grib file
        """
        self.job_name = job_name
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.varslist_json = varslist_json




    def get_varslist(self):
        """
        Function that read varslist_path json file and get variable list
        """
        with open(self.varslist_json) as f:
            self.varslist = json.load(f)
        
        self.varslist_keys = list(self.varslist.keys())
        if not (self.varslist_keys[0] == "surface" or self.varslist_keys[1] == "mutil" ):
            raise ValueError("varslist_json "+ self.varslit_json +  "should have two keys : surface and mutil")
        else:
            self.varslist_surface = self.varslist["surface"]
            self.varslist_mutil = self.varslist["mutil"]
            self.varslist_mutil_vars = self.varslist_mutil.keys()


    @staticmethod
    def source_file_name(year, month, day, hour):
        #src_file = '{:04d}/{:02d}/ecmwf_era5_{:02d}{:02d}{:02d}{:02d}.nc'.format(year, month, year % 100, month, day, hour)
        src_file = 'ecmwf_era5_{:02d}{:02d}{:02d}{:02d}.nc'.format(year % 100, month, day, hour)
        return src_file



    def prepare_era5_data_one_file(self,year,month,date,hour): #extract 2t,tcc,msl,t850,10u,10v
        """
        Process one grib file from source directory  (extract variables and interplolate variable)  and save to output_directory
        args:
            year        : str, the target year to be processed "2017"
            month       : str, the target month to be processed, e.g."01","02","03" ...,"12"
            date        : str, the target date to be processed e.g "01","02","03",..."31"
            hour        : str, the target hour to be processed e.g. "00","01",...,"23"
            varslist_path: str, the path to variable list json file
            output_path : str, the path to output directory
    
        """ 
        temp_path = os.path.join(self.target_dir, year)
        os.makedirs(temp_path, exist_ok=True)
        temp_path = os.path.join(self.target_dir, year, month)
        os.makedirs(temp_path, exist_ok=True)
        
        for var,value in self.varslist_surface.items():
            # surface variables
            infile = os.path.join(self.src_dir, year, month, year+month+date+hour+'_sf.grb')
            outfile = os.path.join(self.target_dir, year, month, year+month+date+hour+'_sfvar.grb')
            outfile_sf = os.path.join(self.target_dir, year, month, year+month+date+hour+'_'+var+'.nc')
            os.system('cdo selname,%s %s %s' % (value,infile,outfile))
            os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
            os.system('rm %s' % outfile)
        

        # multi-level variables
        for var, pl_dic in self.varslist_mutil.items():
            for pl,pl_value in pl_dic.items():
                infile = os.path.join(self.src_dir,year,month,year+month+date+hour+'_ml.grb')
                outfile = os.path.join(self.target_dir,year,month,year+month+date+hour+'_mlvar.grb')
                outfile_sf = os.path.join(self.target_dir,year,month,year+month+date+hour+'_'+var + str(pl_value) +'.nc')
                os.system('cdo -selname,%s -ml2pl,%d %s %s' % (var,pl_value,infile,outfile)) 
                os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
                os.system('rm %s' % outfile)
        
        # merge both variables
        infile = os.path.join(self.target_dir,year,month,year+month+date+hour+'*.nc')
        outfile = os.path.join(self.target_dir,year,month,'ecmwf_era5_'+year[2:]+month+date+hour+'.nc') # change the output file name
        os.system('cdo merge %s %s' % (infile,outfile))
        os.system('rm %s' % (infile))







def extract_time_from_file_name(src_file):
    year = int("20" + src_file[11:13])
    month = int(src_file[13:15])
    day = int(src_file[15:17])
    hour = int(src_file[17:19])
    return year, month, day, hour

def process_era5_in_dir(job_name,src_dir,target_dir):
    print ("job_name",job_name)
    directory_to_process = os.path.join(src_dir, job_name)
    print("Going to process file in directory {}".format(directory_to_process))
    files = os.listdir(directory_to_process)
    os.chdir(directory_to_process)
    #create a subdirectory based on months
    target_dir2 = os.path.join(target_dir,job_name)
    print("The processed files are going to be saved to directory {}".format(target_dir2))
    if not os.path.exists(target_dir2): os.makedirs(target_dir2, exist_ok=True)
    for src_file in files:
        if src_file.endswith(".nc"):
            if os.path.exists(os.path.join(target_dir2, src_file)):
                print(src_file," file has been processed in directory ", target_dir2)
            else:
                print ("==========Processing file {} =============== ".format(src_file))
                prepare_era5_data_one_file(src_file=src_file,directory_to_process=directory_to_process, target=src_file, target_dir=target_dir2)
    # here the defeinition of the failure, success is placed  0=success / -1= fatal-failure / +1 = non-fatal -failure 
    worker_status = 0
    return worker_status
