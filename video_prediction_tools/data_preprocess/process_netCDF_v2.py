'''i
Code for processing staged ERA5 data, this is used for the DataPreprocessing stage of workflow 
'''

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"

import os
import glob
from netCDF4 import Dataset,num2date
import numpy as np
import json
import pickle
import fnmatch
from statistics import Calc_data_stat
import numpy as np
import json

class PreprocessNcToPkl():

    def __init__(self,src_dir=None,target_dir=None,job_name=None,slices=None,vars=("T2","MSL","gph500")):
        '''
        :param src_dir: string, directory based on year  where netCDF-files are stored to be processed
        :param target_dir: directory where pickle-files will be stored
        :param job_name: string "01"-"12" with, job_id passed and organized by PyStager, job_name also corresponds to the month
        :param slices: dictionary e.g.  {'lat_e': 202, 'lat_s': 74, 'lon_e': 710, 'lon_s': 550}, indices defining geographical region of interest
        :param vars: variables to be processed
        '''
        #directory_to_process is month-based directory
        if int(job_name) >12 or int(job_name) < 1 or not isinstance(job_name,str): raise ValueError("job_name should be int type between 1 to 12")
        self.directory_to_process=os.path.join(src_dir, str(job_name)) 
        if not os.path.exists(self.directory_to_process) : raise ("The directory_to_process does not exist")
        self.target_dir = target_dir
        if not os.path.exists(self.target_dir): os.mkdir(self.target_dir)
        self.job_name = job_name
        self.slices = slices
        #target file name need to be saved
        self.target_file = os.path.join(self.target_dir, 'X_' + str(self.job_name) + '.pkl')
        self.vars = vars

    def __call__(self):
       """
       Process the necCDF files in the month_base folder, store the variables of the images into list, store temporal information to list and save them to pickle file 
       """
       if os.path.exists(self.target_file):
         print(target_file," file exists in the directory ", self.target_dir)
       else:
         print ("==========Processing files in directory {} =============== ".format(self.directory_to_process))
         self.get_images_list()
         self.inita_list_and_stat()
         self.process_images_to_list_by_month()
         self.save_images_to_list_by_month() 
         self.save_stat_info() 
         self.save_temp_to_list_by_month()
      
   
    def get_images_list(self):
        """
        Get the images list from the directory_to_process and sort them by date names
        """
        self.imageList_total = list(os.walk(self.directory_to_process, topdown = False))[-1][-1]
        self.filter_not_match_pattern_files()
        self.imageList = sorted(self.imageList)
        return self.imageList
   
    def filter_not_match_pattern_files(self):
        """
        filter the names of netcdf files with the patterns, if any file does not match the file pattern will removed from the imageList
        for the pattern symbol: ^ match start at beginning of the string,[0-9] match a single character in the range 0-9; +matches one or more of the preceding character (greedy match); $ match start at end of the string
         file example :ecmwf_era5_17010219.nc
         """
        patt = "ecmwf_era5_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].nc"
        #self.imageList = self.imageList_total
        #self.imageList = [fnmatch.fnmatch(n, patt) for n in self.imageList_total]
        self.imageList = fnmatch.filter(self.imageList_total,patt)
        return self.imageList


    
    def initia_list_and_stat(self):
        """
        Inits the empty list for store the images and tempral information, and intialise the 
        """
        self.EU_stack_list = [0] * (len(self.imageList))
        self.temporal_list = [0] * (len(self.imageList))
        self.nvars =  len(self.vars)
        self.stat_obj = Calc_data_stat(self.nvars)

    def process_images_to_list_by_month(self):
        """
        Get the selected variables from netCDF file, and concanate all the variables from all the images in the directiory_to_process into a list EU_stack_list
        EU_stack_list dimension should be [numer_of_images,height, width,number_of_variables] 
        temporal_list is 1-dim list with timestamp data type, contains all the timestamps of netCDF files.
        """
        counter = 0 
        for j, im_file in enumerate(self.imageList):
            try:
                im_path = os.path.join(self.directory_to_process, im_file)
                vars_list = []
                with Dataset(im_path,'r') as data_file:
                    times = data_file.variables['time']
                    time = num2date(times[:],units=times.units,calendar=times.calendar)
                    for i in range(self.nvars):
                        var1 = data_file.variables[self.vars[i]][0,self.slices["lat_s"]:self.slices["lat_e"], self.slices["lon_s"]:self.slices["lon_e"]]
                        self.stat_obj.acc_stat_loc(i,var1)
                        vars_list.append(var1)
                EU_stack = np.stack(vars_list, axis=2)
                self.EU_stack_list[j] =list(EU_stack)
                self.temporal_list[j] = list(time)
                print('Open following dataset: '+im_path + "was successfully processed")
            except Exception as err:
                #if the error occurs at the first nc file, we will skip it
                if counter == 0:
                    print("Counter:",counter)
                else:
                    im_path = os.path.join(self.directory_to_process, im_file)
                    print("*************ERROR*************", err)
                    print("Error message {} from file {}".format(err,im_file))
                    self.EU_stack_list[j] = list(EU_stack) # use the previous image as replacement, we can investigate further how to deal with the missing values
                counter += 1
                continue
   


    def save_images_to_list_by_month(self):
        """
        save list of variables from all the images to pickle file
        """
        X = np.array(self.EU_stack_list)
        target_file = os.path.join(self.target_dir, 'X_' + str(job_name) + '.pkl')
        with open(target_file, "wb") as data_file:
            pickle.dump(X,data_file)
        return True



    def save_temp_to_list_by_month(self):
        """
        save the temporal information to pickle file
        """
        temporal_info = np.array(self.temporal_list)
        temporal_file = os.path.join(target_dir, 'T_' + str(job_name) + '.pkl')
        with open(temporal_file,"wb") as ftemp:
            pickle.dump(temporal_info,ftemp)
    
    def save_stat_info(self):
        """
        save the stat information to the target dir
        """
        self.stat_obj.finalize_stat_loc(self.vars)
        self.stat_obj.write_stat_json(self.target_dir,file_id=self.job_name)
       

        
    

 

                
                 
            
            
        
    
    
    
    
    
    



    

