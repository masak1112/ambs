'''
Code for processing staged ERA5 data 
'''

import os
import glob
#import requests
#from bs4 import BeautifulSoup
#import urllib.request
import numpy as np
#from imageio import imread
#from scipy.misc import imresize
from netCDF4 import Dataset
import hickle as hkl
import json

#TODO: Not optimal with DATA_DIR and filingPath: In original process_kitti.py 
# there's just DATA_DIR (which is specified in kitti_settings.py) and in there
# the processed data will be stores. The raw data lies also in there in a subfolder


# ToDo: Define properly the train, val and test index
# Here just for testing and taking weird .DS_Store file into consideration
# http://www.apfelwiki.de/Main/DSStore

# Processes images and saves them in train, val, test splits.
def process_data(directory_to_process, target_dir, job_name, slices, vars=("T2","MSL","gph500")):
    desired_im_sz = (slices["lat_e"] - slices["lat_s"], slices["lon_e"] - slices["lon_s"])
    # ToDo: Define a convenient function to create a list containing all files.
    imageList = list(os.walk(directory_to_process, topdown = False))[-1][-1]
    imageList = sorted(imageList)
    EU_stack_list = [0] * (len(imageList))
    nvars = len(vars)
    #X = np.zeros((len(splits[split]),) + desired_im_sz + (3,), np.uint8)
    #print(X)
    #print('shape of X' + str(X.shape))

    ##### TODO: iterate over split and read every .nc file, cut out array,
    #####		overlay arrays for RGB like style.
    #####		Save everything after for loop.
    EU_stack_list = [0] * (len(imageList))
   
    # ML 2020/04/06 S
    # Some inits
    stat_obj = calc_data_stat(nvars)
    # ML 2020/04/06 E
    for j, im_file in enumerate(imageList):
        try:
            im_path = os.path.join(directory_to_process, im_file)
            print('Open following dataset: '+im_path)

            vars_list = []
            with Dataset(im_path,'r') as data_file:
                for i in range(nvars):
                    var1 = data_file.variables[vars[i]][0,slices["lat_s"]:slices["lat_e"], slices["lon_s"]:slices["lon_e"]]
                    stat_obj.acc_stat_loc(i,var1)
                    vars_list.append(var1)

            EU_stack = np.stack(vars_list, axis = 2)
            EU_stack_list[j] =list(EU_stack)
        except Exception as err:
            print("*************ERROR*************", err)
            print("Error message {} from file {}".format(err,im_file))
            EU_stack_list[j] = list(EU_stack) # use the previous image as replacement, we can investigate further how to deal with the missing values
            continue
            
    X = np.array(EU_stack_list)
    print('Shape of X: ' + str(X.shape))
    target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.hkl')
    hkl.dump(X, target_file) #Not optimal!
    print(target_file, "is saved")
    # ML 2020/03/31: write json file with statistics
    stat_obj.finalize_stat_loc(vars)
    stat_obj.write_stat_json(target_dir,file_id=job_name)

def process_netCDF_in_dir(src_dir,**kwargs):
    target_dir = kwargs.get("target_dir")
    job_name = kwargs.get("job_name")
    directory_to_process = os.path.join(src_dir, job_name)
    os.chdir(directory_to_process)
    if not os.path.exists(target_dir): os.mkdir(target_dir)
    target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.hkl')
    if os.path.exists(target_file):
        print(target_file," file exists in the directory ", target_dir)
    else:
        print ("==========Processing files in directory {} =============== ".format(directory_to_process))
        process_data(directory_to_process=directory_to_process, **kwargs)


def split_data(target_dir, partition= [0.6, 0.2, 0.2]):
    split_dir = target_dir + "/splits"
    if not os.path.exists(split_dir): os.mkdir(split_dir)
    os.chdir(target_dir)
    files = glob.glob("*.hkl")
    filesList = sorted(files)
    # determine correct indicesue
    train_begin = 0
    train_end = round(partition[0] * len(filesList)) - 1
    val_begin = train_end + 1
    val_end = train_end + round(partition[1] * len(filesList))
    test_begin = val_end + 1

    print('Indices of Train, Val and test: ' + str(train_begin) + ' ' + str(val_begin) + ' ' + str(test_begin))
    # slightly adapting start and end because starts at the first index given and stops before(!) the last.
    train_files = filesList[train_begin:val_begin]
    val_files = filesList[val_begin:test_begin]
    test_files = filesList[test_begin:]
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_files
    splits['test'] = test_files
    splits['train'] = train_files
    for split in splits:
        X = []
        files = splits[split]
        for file in files:
            data_file = os.path.join(target_dir,file)
            #load data with hkl file
            data = hkl.load(data_file)
            X = X + list(data)
        X = np.array(X)
        print("==================={}=====================".format(split))
        print ("Sources for {} dataset are {}".format(split,files))
        print("Number of images in {} dataset is {} ".format(split,len(X)))
        print ("dataset shape is {}".format(np.array(X).shape))
        #save training, val and test data into splits directoyr
        hkl.dump(X, os.path.join(split_dir, 'X_' + split + '.hkl'))
        hkl.dump(files, os.path.join(split_dir,'sources_' + split + '.hkl'))

# ML 2020/05/15 S
def get_unique_vars(varnames):
    vars_uni, varsind = np.unique(varnames,return_index = True)
    nvars_uni         = len(vars_uni)
    
    return(vars_uni, varsind, nvars_uni)

class calc_data_stat:
    """Class for computing statistics and saving them to a json-files."""
    def __init__(self,nvars):
        self.stat_dict = {}
        self.varmin    = np.full(nvars,np.nan)
        self.varmax    = np.full(nvars,np.nan)
        self.varavg    = np.zeros((nvars,1))            # second dimension acts as placeholder for averaging on master node collecting json-files from slave nodes
        self.nfiles    = [0]                            # number of processed files
        self.mode      = ""                             # mode to distinguish between processing on slave and master nodes (sanity check)
        self.jsfiles   = [""]                           # list of processed json-files (master-mode only!)

    def acc_stat_loc(self,ivar,data):
        """
        Performs accumulation of all statistics while looping through all data files (i.e. updates the statistics) on slave nodes
        """
        if not self.mode: 
            self.mode = "loc"
        elif self.mode == "master":
            raise ValueError("Cannot switch to loc-mode during runtime...")
        else:
            pass
    
        self.varmin[ivar]    = np.fmin(self.varmin[ivar],np.amin(data))
        self.varmax[ivar]    = np.fmax(self.varmax[ivar],np.amax(data))
        self.varavg[ivar,0] += np.average(data)                           # note that we sum the average -> readjustment required in the final step
        if (ivar == 0): self.nfiles[0] += 1 
        
    def finalize_stat_loc(self,varnames):
        """
        Finalizes computation of statistics after going through all the data on slave nodes.
        Afterwards the statistics dictionary is ready for being written in a json-file
        """
        
        if self.mode != "loc":
            raise ValueError("Object is not in loc-mode. Probably some master-method has been called previously.")
        
        if self.stat_dict: raise ValueError("Statistics dictionary is not empty.")
        
        vars_uni, varsind = np.unique(varnames,return_index=True)
        nvars = len(vars_uni)
        
        vars_uni, varsind, nvars = get_unique_vars(varnames)

        varmin, varmax, varavg = self.varmin[varsind], self.varmax[varsind], self.varavg[varsind,0] 
        
        for i in range(nvars):
            varavg[i] /= self.nfiles                                    # for adjusting the (summed) average

            self.stat_dict[vars_uni[i]]=[]
            self.stat_dict[vars_uni[i]].append({
                  'min': varmin[i],
                  'max': varmax[i],
                  'avg': varavg[i]
            })        
        self.stat_dict["common_stat"] = [
            {"nfiles":self.nfiles[0]}]
        
    def acc_stat_master(self,file_dir,file_id):
        """ Opens statistics-file (created by slave nodes) and accumulates its content."""
       
        if (int(file_id) <= 0): raise ValueError("Non-valid file_id passed.")
      
        if not self.mode: 
            self.mode = "master"
        elif self.mode == "loc":
            raise ValueError("Cannot switch to master-mode during runtime...")
        else:
            pass
        
        # sanity check: check if dictionary is initialized with unique values only
        if self.stat_dict.keys() > set(self.stat_dict.keys()):
            raise ValueError("Initialized dictionary contains duplicates of variales. Need unique collection instead.")
        else:
            pass

        file_name = os.path.join(file_dir,"stat_{0:0=2d}.json".format(int(file_id)))
        
        if not file_name in self.jsfiles:
            print("Try to open: '"+file_name+"'")
            
            try:
                with open(file_name) as js_file:                
                    dict_in = json.load(js_file)
                    
                    # sanity check
                    if (len(dict_in.keys()) -1 != len(self.varmin)):
                        raise ValueError("Different number of variables found in json-file '"+js_file+"' as expected from statistics object.")

                    self.varmin  = np.fmin(self.varmin,calc_data_stat.get_var_stat(dict_in,"min")) 
                    self.varmax  = np.fmax(self.varmax,calc_data_stat.get_var_stat(dict_in,"max"))
                    if (all(self.varavg == 0.) or self.nfiles[0] == 0):
                        self.varavg    = calc_data_stat.get_var_stat(dict_in,"avg")
                        self.nfiles[0] = calc_data_stat.get_common_stat(dict_in,"nfiles")
                        self.jsfiles[0]= file_name    
                    else:
                        self.varavg = np.append(self.varavg,calc_data_stat.get_var_stat(dict_in,"avg"),axis=1)
                        self.nfiles.append(calc_data_stat.get_common_stat(dict_in,"nfiles"))
                        self.jsfiles.append(file_name)
            except IOError:
                print("Cannot handle statistics file '"+file_name+"' to be processed.")
            except ValueError:
                print("Cannot retireve all required statistics from '"+file_name+"'")
        else:
            print("Statistics file '"+file_name+"' has already been processed. Thus, just pass here...")
            pass
            
    def finalize_stat_master(self,path_out,vars_uni):
        """Performs final compuattion of statistics after accumulation from slave nodes."""
        if self.mode != "master":
            raise ValueError("Object is not in master-mode. Probably some loc-method has been called previously.")
        
        if len(vars_uni) > len(set(vars_uni)):
            raise ValueError("Input variable names are not unique.")
                
        js_file = os.path.join(path_out,"statistics.json")
        nvars     = len(vars_uni)
        n_jsfiles = len(self.nfiles)
        nfiles_all= np.sum(self.nfiles)
        avg_wgt   = np.array(self.nfiles,dtype=float)/float(nfiles_all)

        varmin, varmax = self.varmin, self.varmax
        varavg    = np.sum(np.multiply(self.varavg,avg_wgt),axis=1)        # calculate weighted average

        for i in range(nvars):
            self.stat_dict[vars_uni[i]]=[]
            self.stat_dict[vars_uni[i]].append({
                  'min': varmin[i].tolist(),
                  'max': varmax[i].tolist(),
                  'avg': varavg[i].tolist()
            })        
        self.stat_dict["common_stat"] = [
            {"nfiles":int(nfiles_all)}]    
        
    @staticmethod
    def get_var_stat(stat_dict,stat_name):
        '''
        Unpacks statistics dictionary and returns values of stat_name
        '''        
        
        # some sanity checks
        if not stat_dict: raise ValueError("Input dictionary is still empty! Cannot access anything from it.")
        if not "common_stat" in stat_dict.keys(): raise ValueError("Input dictionary does not seem to be a proper statistics dictionary as common_stat-element is missing.")
        
        stat_dict_filter = (stat_dict).copy()
        stat_dict_filter.pop("common_stat")
        
        if not stat_dict_filter.keys(): raise ValueError("Input dictionary does not contain any variables.")
       
        try:
            varstat = np.array([stat_dict_filter[i][0][stat_name] for i in [*stat_dict_filter.keys()]])
            if np.ndim(varstat) == 1:         # avoid returning rank 1-arrays
                return varstat.reshape(-1,1)
            else:
                return varstat
        except:
            raise ValueError("Could not find "+stat_name+" for all variables of input dictionary.")       
        
    @staticmethod    
    def get_stat_allvars(stat_dict,stat_name,allvars):
        '''
        Retrieves requested statistics (stat_name) for all variables listed in allvars given statistics dictionary.
        '''        
        
        if not stat_dict: raise ValueError("Statistics dictionary is still empty! Cannot access anything from it.")
        if not "common_stat" in stat_dict.keys(): raise ValueError("Input dictionary does not seem to be a proper statistics dictionary as common_stat-element is missing.")    
    
        vars_uni,indrev = np.unique(allvars,return_inverse=True)
    
        try:
            return([stat_dict[var][0][stat_name] for var in vars_uni[indrev]]) 
        except:
            raise ValueError("Could not find "+stat_name+" for all variables of input dictionary.")
    
    @staticmethod
    def get_common_stat(stat_dict,stat_name):
        
        if not stat_dict: raise ValueError("Input dictionary is still empty! Cannot access anything from it.")
        if not "common_stat" in stat_dict.keys(): raise ValueError("Input dictionary does not seem to be a proper statistics dictionary as common_stat-element is missing.")
        
        common_stat_dict = stat_dict["common_stat"][0]
        
        try:
            return(common_stat_dict[stat_name])
        except:
            raise ValueError("Could not find "+stat_name+" in common_stat of input dictionary.")
        
        
    def write_stat_json(self,path_out,file_id = -1):
        """
        Writes statistics-dictionary of slave nodes to json-file (with job_id in the output name)
        If file_id is passed (and greater than 0), parallelized peration on a slave node is assumed.
        Else: method is invoked from master node, i.e. final json-file is created
        """
        if (self.mode == "loc"):
            if int(file_id) <= 0: raise ValueError("Object is in loc-mode, but no valid file_id passed")
            # json-file from slave node
            js_file = os.path.join(path_out,'stat_{0:0=2d}.json'.format(int(file_id)))
        elif (self.mode == "master"):
            if (int(file_id) > 0): print("Warning: Object is master-mode, but file_id passed which will be ignored.")
            # (final) json-file from master node 
            js_file = os.path.join(path_out,'statistics.json')
        else:
            raise ValueError("Object seems to be initialized only, but no data has been processed so far.")
       
        try:
            with open(js_file,'w') as stat_out:
                json.dump(self.stat_dict,stat_out)
        except ValueError: 
            print("Something went wrong when writing dictionary to json-file: '"+js_file+"''")
        finally:
            print("Created statistics json-file '"+js_file+"' successfully.")

# ML 2020/05/15 E
                 

def split_data_multiple_years(target_dir,partition,varnames):
    """
    Collect all the X_*.hkl data across years and split them to training, val and testing datatset
    """

    #target_dirs = [os.path.join(target_dir,year) for year in years]
    #os.chdir(target_dir)
    splits_dir = os.path.join(target_dir,"splits")
    os.makedirs(splits_dir, exist_ok=True) 
    splits = {s: [] for s in list(partition.keys())}
    # ML 2020/05/19 S
    vars_uni, varsind, nvars = get_unique_vars(varnames)
    stat_obj = calc_data_stat(nvars)
    
    for split in partition.keys():
        values = partition[split]
        files = []
        X = []
        for year in values.keys():
            file_dir = os.path.join(target_dir,year)
            for month in values[year]:
                month = "{0:0=2d}".format(month)
                hickle_file = "X_{}.hkl".format(month)
                data_file = os.path.join(file_dir,hickle_file)
                files.append(data_file)
                data = hkl.load(data_file)
                X = X + list(data)
                # process stat-file:
                stat_obj.acc_stat_master(file_dir,int(month))
                
        X = np.array(X) 
        print("==================={}=====================".format(split))
        print ("Sources for {} dataset are {}".format(split,files))
        print("Number of images in {} dataset is {} ".format(split,len(X)))
        print ("dataset shape is {}".format(np.array(X).shape))
        hkl.dump(X, os.path.join(splits_dir , 'X_' + split + '.hkl'))
        hkl.dump(files, os.path.join(splits_dir,'sources_' + split + '.hkl'))
        
    # write final statistics json-file
    stat_obj.finalize_stat_master(target_dir,vars_uni)
    stat_obj.write_stat_json(target_dir)
        

        
    

                
            
                
            
            
        
    
    
    
    
    
    



    

