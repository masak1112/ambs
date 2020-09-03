""" 
Class to retrieve and handle meta-data
"""

import os
import sys
import time
import numpy as np
import json
from netCDF4 import Dataset
from helper import is_integer, add_str_to_path

class MetaData:
    """
     Class for handling, storing and retrieving meta-data
    """
    
    def __init__(self,json_file=None,suffix_indir=None,data_filename=None,slices=None,variables=None):
        
        """
         Initailizes MetaData instance by reading a corresponding json-file or by handling arguments of the Preprocessing step
         (i.e. exemplary input file, slices defining region of interest, input variables) 
        """
        
        method_name = MetaData.__init__.__name__+" of Class "+MetaData.__name__
        
        if not json_file is None: 
            print(json_file)
            print(type(json_file))
            MetaData.get_metadata_from_file(self,json_file)
            
        else:
            # No dictionary from json-file available, all other arguments have to set
            if not suffix_indir:
                raise TypeError(method_name+": 'suffix_indir'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(suffix_indir,str):
                    raise TypeError(method_name+": 'suffix_indir'-argument must be a string.")
            
            if not data_filename:
                raise TypeError(method_name+": 'data_filename'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(data_filename,str):
                    raise TypeError(method_name+": 'data_filename'-argument must be a string.")
                
            if not slices:
                raise TypeError(method_name+": 'slices'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(slices,dict):
                    raise TypeError(method_name+": 'slices'-argument must be a dictionary.")
            
            if not variables:
                raise TypeError(method_name+": 'variables'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(variables,list):
                    raise TypeError(method_name+": 'variables'-argument must be a list.")       
            
            MetaData.get_and_set_metadata_from_file(self,suffix_indir,data_filename,slices,variables)
            
            MetaData.write_metadata_to_file(self)
            

    def get_and_set_metadata_from_file(self,suffix_indir,datafile_name,slices,variables):
        """
         Retrieves several meta data from netCDF-file and sets corresponding class instance attributes.
         Besides, the name of the experiment directory is constructed following the naming convention (see below)
        
         Naming convention:
         [model_base]_Y[yyyy]to[yyyy]M[mm]to[mm]-[nx]x[ny]-[nnnn]N[eeee]E-[var1]_[var2]_(...)_[varN]
         ---------------- Given ----------------|---------------- Created dynamically --------------
        
         Note that the model-base as well as the date-identifiers must already be included in target_dir_in.
        """
        
        method_name = MetaData.get_and_set_metadata_from_file.__name__+" of Class "+MetaData.__name__
        
        if not suffix_indir: raise ValueError(method_name+": suffix_indir must be a non-empty path.")
    
        # retrieve required information from file 
        flag_coords = ["N", "E"]
 
        print("Retrieve metadata based on file: '"+datafile_name+"'")
        try:
            datafile = Dataset(datafile_name,'r')
        except:
            print(method_name + ": Error when handling data file: '"+datafile_name+"'.")
            exit()
        
        # Check if all requested variables can be obtained from datafile
        MetaData.check_datafile(datafile,variables)
        self.varnames    = variables
        
        
        self.nx, self.ny = np.abs(slices['lon_e'] - slices['lon_s']), np.abs(slices['lat_e'] - slices['lat_s'])    
        sw_c             = [float(datafile.variables['lat'][slices['lat_e']-1]),float(datafile.variables['lon'][slices['lon_s']])]                # meridional axis lat is oriented from north to south (i.e. monotonically decreasing)
        self.sw_c        = sw_c
        self.lat = datafile.variables['lat'][slices['lat_s']:slices['lat_e']]
        self.lon = datafile.variables['lon'][slices['lon_s']:slices['lon_e']]
        
        # Now start constructing expdir-string
        # switch sign and coordinate-flags to avoid negative values appearing in expdir-name
        if sw_c[0] < 0.:
            sw_c[0] = np.abs(sw_c[0])
            flag_coords[0] = "S"
        if sw_c[1] < 0.:
            sw_c[1] = np.abs(sw_c[1])
            flag_coords[1] = "W"
        nvar     = len(variables)
        
        # splitting has to be done in order to retrieve the expname-suffix (and the year if required)
        path_parts = os.path.split(suffix_indir.rstrip("/"))
        
        if (is_integer(path_parts[1])):
            year = path_parts[1]
            path_parts = os.path.split(path_parts[0].rstrip("/"))
        else:
            year = ""
        
        expdir, expname = path_parts[0], path_parts[1] 

        # extend expdir_in successively (splitted up for better readability)
        expname += "-"+str(self.nx) + "x" + str(self.ny)
        expname += "-"+(("{0: 05.2f}"+flag_coords[0]+"{1:05.2f}"+flag_coords[1]).format(*sw_c)).strip().replace(".","")+"-"  
        
        # reduced for-loop length as last variable-name is not followed by an underscore (see above)
        for i in range(nvar-1):
            expname += variables[i]+"_"
        expname += variables[nvar-1]
        
        self.expname = expname
        self.expdir  = expdir
        self.status  = ""                   # uninitialized (is set when metadata is written/compared to/with json-file, see write_metadata_to_file-method)

    # ML 2020/04/24 E         
    
    def write_metadata_to_file(self,dest_dir = None):
        
        """
         Write meta data attributes of class instance to json-file.
        """
        
        method_name = MetaData.write_metadata_to_file.__name__+" of Class "+MetaData.__name__
        # actual work:
        meta_dict = {"expname": self.expname,
                     "expdir" : self.expdir}
        
        meta_dict["sw_corner_frame"] = {
            "lat" : np.around(self.sw_c[0],decimals=2),
            "lon" : np.around(self.sw_c[1],decimals=2)
            }
        
        meta_dict["coordinates"] = {
            "lat" : np.around(self.lat,decimals=2).tolist(),
            "lon" : np.around(self.lon,decimals=2).tolist()
            }
            
        meta_dict["frame_size"] = {
            "nx" : int(self.nx),
            "ny" : int(self.ny)
            }
        
        meta_dict["variables"] = []
        for i in range(len(self.varnames)):
            #print(self.varnames[i])
            meta_dict["variables"].append( 
                    {"var"+str(i+1) : self.varnames[i]})
        
        # create directory if required
        if dest_dir is None: 
            dest_dir = os.path.join(self.expdir,self.expname)
        if not os.path.exists(dest_dir):
            print("Created experiment directory: '"+self.expdir+"'")
            os.makedirs(dest_dir,exist_ok=True)            
            
        meta_fname = os.path.join(dest_dir,"metadata.json")

        if os.path.exists(meta_fname):                      # check if a metadata-file already exists and check its content
            print(method_name+": json-file ('"+meta_fname+"' already exists. Its content will be checked...")
            self.status = "old"                             # set status to old in order to prevent repeated modification of shell-/Batch-scripts
            with open(meta_fname,'r') as js_file:
                dict_dupl = json.load(js_file)
                
                if dict_dupl != meta_dict:
                    meta_fname_dbg = os.path.join(dest_dir,"metadata_debug.json")
                    print(method_name+": Already existing metadata (see '"+meta_fname+"') do not fit data being processed right now (see '" \
                          +meta_fname_dbg+"'. Ensure a common data base.")
                    with open(meta_fname_dbg,'w') as js_file:
                        json.dump(meta_dict,js_file)                         
                    raise ValueError
                else: #do not need to do anything
                    pass
        else:
            # write dictionary to file
            print(method_name+": Write dictionary to json-file: '"+meta_fname+"'")
            with open(meta_fname,'w') as js_file:
                json.dump(meta_dict,js_file)
            self.status = "new"                             # set status to new in order to trigger modification of shell-/Batch-scripts
        
    def get_metadata_from_file(self,js_file):
        
        """
         Retrieves meta data attributes from json-file
        """
        
        with open(js_file) as js_file:                
            dict_in = json.load(js_file)
            
            self.expdir = dict_in["expdir"]
            
            self.sw_c       = [dict_in["sw_corner_frame"]["lat"],dict_in["sw_corner_frame"]["lon"] ]
            self.lat        = dict_in["coordinates"]["lat"]
            self.lon        = dict_in["coordinates"]["lon"]
            
            self.nx         = dict_in["frame_size"]["nx"]
            self.ny         = dict_in["frame_size"]["ny"]
            # dict_in["variables"] is a list like [{var1: varname1},{var2: varname2},...]
            list_of_dict_aux = dict_in["variables"] 
            # iterate through the list with an integer ivar
            # note: the naming of the variables starts with var1, thus add 1 to the iterator
            self.variables = [list_of_dict_aux[ivar]["var"+str(ivar+1)] for ivar in range(len(list_of_dict_aux))]
            
    def write_dirs_to_batch_scripts(self,batch_script):
        
        """
         Expands ('known') directory-variables in batch_script by expdir-attribute of class instance
        """
        
        paths_to_mod = ["source_dir=","destination_dir=","checkpoint_dir=","results_dir="]      # known directory-variables in batch-scripts
        
        with open(batch_script,'r') as file:
            data = file.readlines()
            
        nlines = len(data)
        matched_lines = [iline for iline in range(nlines) if any(str_id in data[iline] for str_id in paths_to_mod)]   # list of line-number indices to be modified 

        for i in matched_lines:
            data[i] = add_str_to_path(data[i],self.expname)

        
        with open(batch_script,'w') as file:
            file.writelines(data)
    
    @staticmethod
    def write_destdir_jsontmp(dest_dir, tmp_dir = None):        
        """
          Writes dest_dir to temporary json-file (temp.json) stored in the current working directory.
          To be executed by Master node in parallel mode.
        """
        
        if not tmp_dir: tmp_dir = os.getcwd()
        
        file_tmp = os.path.join(tmp_dir,"temp.json")
        dict_tmp = {"destination_dir": dest_dir}
        
        with open(file_tmp,"w") as js_file:
            print("Save destination_dir-variable in temporary json-file: '"+file_tmp+"'")
            json.dump(dict_tmp,js_file)
            
    @staticmethod
    def get_destdir_jsontmp(tmp_dir = None):
        """
          Retrieves dest_dir from temporary json-file which is expected to exist in the current working directory and returns it.
        """
        
        method_name = MetaData.get_destdir_jsontmp.__name__+" of Class "+MetaData.__name__

        if not tmp_dir: tmp_dir = os.getcwd()
        
        file_tmp = os.path.join(tmp_dir,"temp.json")
        
        try:
            with open(file_tmp,"r") as js_file:
                dict_tmp = json.load(js_file)
        except:
            print(method_name+": Could not open requested json-file '"+file_tmp+"'")
            sys.exit(1)
            
        if not "destination_dir" in dict_tmp.keys():
            raise Exception(method_name+": Could not find 'destination_dir' in dictionary obtained from "+file_tmp)
        else:
            return(dict_tmp.get("destination_dir"))
    
    @staticmethod
    def wait_for_jsontmp(tmp_dir = None, waittime = 10, delay=0.5):
        """
          Waits at max. waittime (in sec) until temp.json-file becomes available
        """
        
        method_name = MetaData.wait_for_jsontmp.__name__+" of Class "+MetaData.__name__
        
        if not tmp_dir: tmp_dir = os.getcwd()
        
        file_tmp = os.path.join(tmp_dir,"temp.json")
                                
        counter_max = waittime/delay
        counter = 0
        status  = "not_ok" 
        
        while (counter <= counter_max):
            if os.path.isfile(file_tmp):
                status = "ok"
                break
            else:
                time.sleep(delay)
            
            counter += 1
                                
        if status != "ok": raise IOError(method_name+": '"+file_tmp+ \
                           "' does not exist after waiting for "+str(waittime)+" sec.") 
    
    
    @staticmethod
    def issubset(a,b):
        """
        Checks if all elements of a exist in b or vice versa (depends on the length of the corresponding lists/sets)
        """  
        
        if len(a) > len(b):
            return(set(b).issubset(set(a)))
        elif len(b) >= len(a):
            return(set(a).issubset(set(b)))
    
    @staticmethod
    def check_datafile(datafile,varnames):
        """
          Checks if all varnames can be found in datafile
        """
        
        if not MetaData.issubset(varnames,datafile.variables.keys()):
            for i in range(len(varnames2check)):
                if not varnames2check[i] in f0.variables.keys():
                    print("Variable '"+varnames2check[i]+"' not found in datafile '"+data_filenames[0]+"'.")
                raise ValueError("Could not find the above mentioned variables.")
        else:
            pass


        
# ----------------------------------- end of class MetaData -----------------------------------

                       
    
        
        
