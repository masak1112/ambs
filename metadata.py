""" 
Classes and routines to retrieve and handle meta-data
"""

import os
import numpy as np
import json
from netCDF4 import Dataset

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
            MetaData.get_metadata_from_file(json_file)
            
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
        
        method_name = MetaData.__init__.__name__+" of Class "+MetaData.__name__
        
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
        
        # Now start constructing exp_dir-string
        # switch sign and coordinate-flags to avoid negative values appearing in exp_dir-name
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

        # extend exp_dir_in successively (splitted up for better readability)
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
        
        method_name = MetaData.__init__.__name__+" of Class "+MetaData.__name__
        # actual work:
        meta_dict = {"expname": self.expname,
                     "expdir" : self.exp_dir}
        
        meta_dict["sw_corner_frame"] = {
            "lat" : self.sw_c[0],
            "lon" : self.sw_c[1]
            }
        
        meta_dict["frame_size"] = {
            "nx" : int(self.nx),
            "ny" : int(self.ny)
            }
        
        meta_dict["variables"] = []
        for i in range(len(self.varnames)):
            print(self.varnames[i])
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
            self.status = "old"                             # set status to old in order to prevent repeated modification of shell-/Batch-scripts
            with open(meta_fname,'r') as js_file:
                dict_dupl = json.load(js_file)
                
                if dict_dupl != meta_dict:
                    print(method_name+": Already existing metadata (see '"+meta_fname+") do not fit data being processed right now. Ensure a common data base.")
                    sys.exit(1)
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
            
            self.exp_dir = dict_in["exp_dir"]
            
            self.sw_c       = [dict_in["sw_corner_frame"]["lat"],dict_in["sw_corner_frame"]["lon"] ]
            
            self.nx         = dict_in["frame_size"]["nx"]
            self.ny         = dict_in["frame_size"]["ny"]
            
            self.variables  = [dict_in["variables"][ivar] for ivar in dict_in["variables"].keys()]   
            
            
    def write_dirs_to_batch_scripts(self,batch_script):
        
        """
         Expands ('known') directory-variables in batch_script by exp_dir-attribute of class instance
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
    def write_destdir_jsontmp(dest_dir):        
        """
          Writes dest_dir to temporary json-file (temp.json) stored in the current working directory.
        """
        
        file_tmp = os.path.join(os.getcwd(),"temp.json")
        dict_tmp = {"destination_dir": dest_dir}
        
        with open(file_tmp,"w") as js_file:
            print("Save destination_dir-variable in temporary json-file: '"+file_tmp+"'")
            json.dump(dict_tmp,js_file)
            
    @staticmethod
    def get_destdir_jsontmp():
        """
          Retrieves dest_dir from temporary json-file which is expected to exist in the current working directory and returns it.
        """
        
        file_tmp = os.path.join(os.getcwd(),"temp.json")
        
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

# some auxilary functions which are not bound to MetaData-class 

def add_str_to_path(path_in,add_str):
    
    """
        Adds add_str to path_in if path_in does not already end with add_str.
        Function is also capable to handle carriage returns for handling input-strings obtained by reading a file.
    """
    
    l_linebreak = path_in.endswith("\n")   # flag for carriage return at the end of input string
    line_str    = path_in.rstrip("\n")
    
    if (not line_str.endswith(add_str)) or \
       (not line_str.endswith(add_str.rstrip("/"))):
        
        line_str = line_str + add_str + "/"
    else:
        print(add_str+" is already part of "+line_str+". No change is performed.")
    
    if l_linebreak:                     # re-add carriage return to string if required
        return(line_str+"\n")
    else:
        return(line_str)
                       
                       
def is_integer(n):
    '''
    Checks if input string is numeric and of type integer.
    '''
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()                       
                       
    
        
        
