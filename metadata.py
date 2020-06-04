""" 
Classes and routines to retrieve and handle meta-data
"""

import numpy as np
import json
from netCDF4 import Dataset

class MetaData:
    """
        Class for handling, storing and retrieving meta-data
    """
    
    def __init__(self,json_file=None,suffix_indir=None,data_filename=None,slices=None,variables=None):
        
        method_name = MetaData.__init__.__name__" of Class "+MetaData.__name__
        
        if not json_file: 
            
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
                
            MetaData.get_and_set_metadata_from_file(suffix_indir,data_filename,slices,variables)
            
            MetaData.write_metadata_to_file()
            

    def get_and_set_metadata_from_file(self,suffix_indir,datafile_name,slices,variables):
        """
        Construct path to target directory following naming convention.
        Note that the path to the absolute directory must be passed via the target_dir_in-argument.
        This path will be expanded by a string completing the input argument.
        
        Naming convention:
        [model_base]_Y[yyyy]to[yyyy]M[mm]to[mm]-[nx]x[ny]-[nnnn]N[eeee]E-[var1]_[var2]_(...)_[varN]
        ---------------- Given ----------------|---------------- Created dynamically --------------
        
        Note that the model-base as well as the date-identifiers must already be included in target_dir_in.
        """
        
        if not suffix_indir: raise ValueError("suffix_indir must be a non-empty path.")
    
        # retrieve required information from file 
        flag_coords = ["N", "E"]
 
        print("Retrieve metadata based on file: '"+datafile_name+"'")
        datafile = Dataset(datafile_name,'r')
        
        # Check if all requested variables can be obtained from datafile
        MetaData.check_datafile(datafile,variables)
        self.varnames    = variables
        
        
        self.nx, self.ny = np.abs(slices['lon_e'] - slices['lon_s']), np.abs(slices['lat_e'] - slices['lat_s'])    
        sw_c             = [datafile.variables['lat'][slices['lat_e']-1],datafile.variables['lon'][slices['lon_s']]]                # meridional axis lat is oriented from north to south (i.e. monotonically decreasing)
        self.sw_c        = sw_c
        
        # Now start constructing target_dir-string
        # switch sign and coordinate-flags to avoid negative values appearing in target_dir-name
        if sw_c[0] < 0.:
            sw_c[0] = np.abs(sw_c[0])
            flag_coords[0] = "S"
        if sw_c[1] < 0.:
            sw_c[1] = np.abs(sw_c[1])
            flag_coords[1] = "W"
        nvar     = len(variables)
        
        # splitting has to be done in order to avoid the occurence of the year-identifier in the target_dir-path
        path_parts = os.path.split(suffix_indir.rstrip("/"))
        
        if (is_integer(path_parts[1])):
            target_dir = path_parts[0]
            year = path_parts[1]
        else:
            target_dir = suffix_indir
            year = ""

        # extend target_dir_in successively (splitted up for better readability)
        target_dir += "-"+str(nx) + "x" + str(ny)
        target_dir += "-"+(("{0: 06.2f}"+flag_coords[0]+"{1: 06.2f}"+flag_coords[1]).format(*sw_c)).strip().replace(".","")+"-"  
        
        # reduced for-loop length as last variable-name is not followed by an underscore (see above)
        for i in range(nvar-1):
            target_dir += variables[i]+"_"
        target_dir += variables[nvar-1]
        
        self.target_dir = target_dir

    # ML 2020/04/24 E 
    
    def write_metadata_to_file(self):
        
        meta_dict = {"target_dir": self.target_dir}
        
        meta_dict["sw_corner_frame"] = {
            "lat" : self.sw_c[0]
            "lon" : self.sw_c[1]
            }
        
        meta_dict["frame_size"] = {
            "nx" = self.nx
            "ny" = self.ny
            }
        
        meta_dict["variables"] = []
        for i in range(len(self.varnames)):
            meta_dict["variables"]["var"+str(i+1)] = self.varnames[i]
            
        meta_fname = os.path.join(self.target_dir,"metadata.json")
        
        # write dictionary to file
        with open(meta_fname) as js_file:
            json.dump(js_file,meta_dict)
            
    def get_metadata_from_file(self,js_file):
        
        with open(js_file) as js_file:                
            dict_in = json.load(js_file)
            
            self.target_dir = dict_in["target_dir"]
            
            self.sw_c       = [dict_in["sw_corner_frame"]["lat"],dict_in["sw_corner_frame"]["lon"] ]
            
            self.nx         = dict_in["frame_size"]["nx"]
            self.ny         = dict_in["frame_size"]["ny"]
            
            self.variables  = [dict_in["variables"][ivar] for ivar in dict_in["variables"].keys()]   
    
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
        
        
    
                       
                       
                       
    
        
        
