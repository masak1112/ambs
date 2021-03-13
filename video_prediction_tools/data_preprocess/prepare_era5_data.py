"""
Functions required for extracting ERA5 data.
"""
import os
import json
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler, Michael Langguth,Yanji"
__date__ = "unknown"
# specify source and target directories


class ERA5DataExtraction(object):

    def __init__(self,year,job_name,src_dir,target_dir,varslist_json):
        """
        Function to extract ERA5 data from slmet 
        args:
             year        : str, the target year to be processed "2017"
             job_name     :int from 1 to 12 correspoding to month
             scr_dir      :str, upper level of directory at year level
             target_dir   : str, upper level of directory at year level
             varslist_json: str, the path to the varibale list that to be extracted from original grib file
        """
        self.year = year
        self.job_name = job_name
        self.src_dir = src_dir
        self.target_dir = target_dir
        self.varslist_json = varslist_json
        self.get_varslist()

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


    def prepare_era5_data_one_file(self, month, day, hour): #extract 2t,tcc,msl,t850,10u,10v
        """
        Process one grib file from source directory  (extract variables and interplolate variable)  and save to output_directory
        args:
            month       : str, the target month to be processed, e.g."01","02","03" ...,"12"
            date        : str, the target date to be processed e.g "01","02","03",..."31"
            hour        : str, the target hour to be processed e.g. "00","01",...,"23"
            varslist_path: str, the path to variable list json file
            output_path : str, the path to output directory
    
        """ 
        temp_path = os.path.join(self.target_dir, self.year)
        os.makedirs(temp_path, exist_ok=True)
        temp_path = os.path.join(self.target_dir, self.year, month)
        os.makedirs(temp_path, exist_ok=True)
        
        for var,value in self.varslist_surface.items():
            # surface variables
            infile = os.path.join(self.src_dir, self.year, month, self.year+month+day+hour+'_sf.grb')
            outfile_sf = os.path.join(self.target_dir, self.year, month, self.year+month+day+hour+'_'+var+'.nc')
            os.system('cdo -f nc copy -selname,%s %s %s' % (value,infile,outfile_sf))
            os.system('cdo -chname,%s,%s %s %s' % (var,value,outfile_sf,outfile_sf)) 

        # multi-level variables
        for var, pl_dic in self.varslist_mutil.items():
            for pl, pl_value in pl_dic.items():
                infile = os.path.join(self.src_dir, self.year, month, self.year+month+date+hour+'_ml.grb')
                outfile_sf = os.path.join(self.target_dir, self.year, month, self.year+month+date+hour+'_'+var + str(pl_value) +'.nc')
                os.system('cdo -f nc copy -selname,%s -ml2pl,%d %s %s' % (var,pl_value,infile,outfile_sf)) 
                os.system('cdo -chname,%s,%s %s %s' % (var,var+"_"+str(pl_value),outfile_sf,outfile_sf))           
        # merge both variables
        infile = os.path.join(self.target_dir, self.year, month, self.year+month+date+hour+'*.nc')
        outfile = os.path.join(self.target_dir, self.year, month, 'ecmwf_era5_'+self.year[2:]+month+date+hour+'.nc') # change the output file name
        os.system('cdo merge %s %s' % (infile,outfile))
        os.system('rm %s' % (infile))




    def process_era5_in_dir(self):
        """
        Function that extract data at year level
        """
        
        dates = list(range(1,32))
        dates = ["{:02d}".format(d) for d in dates]

        hours = list(range(0,24))
        hours = ["{:02d}".format(h) for h in hours]
       
        print ("job_name",self.job_name)
        for d in dates:
            for h in hours:
                self.prepare_era5_data_one_file(self.job_name,d,h)
                    # here the defeinition of the failure, success is placed  0=success / -1= fatal-failure / +1 = non-fatal -failure 
        worker_status = 0
        return worker_status
