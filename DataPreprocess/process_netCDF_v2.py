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
import copy

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
                    stat_obj.acc_stat(i,var1)
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
    stat_obj.write_stat_json_loc(target_dir,job_name)

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

# ML 2020/04/03 S
def get_stat(stat_dict,stat_name):
    '''
    Unpacks statistics dictionary and returns values of stat_name
    '''
    if ("common_stat" in stat_dict):
        # remove dictionary elements not related to specific variables, i.e. common_stat-elements
        stat_dict_filter = copy.deepcopy(stat_dict)
        stat_dict_filter.pop("common_stat")
    else:
        stat_dict_filter = stat_dict
    
    try:
        return [stat_dict_filter[i][0][stat_name] for i in [*stat_dict_filter.keys()]]
    except:
        raise ValueError("Could not find "+stat_name+" for all variables of input dictionary.")

# ML 2020/04/13 S
def get_stat_allvars(stat_dict,stat_name,allvars):
    '''
    Retrieves requested statistics (stat_name) for all variables listed in allvars given statistics dictionary.
    '''
    vars_uni,indrev = np.unique(allvars,return_inverse=True)
    
    try:
        return([stat_dict[var][0][stat_name] for var in vars_uni[indrev]]) 
    except:
        raise ValueError("Could not find "+stat_name+" for all variables of input dictionary.")

# ML 2020/04/13: E

def create_stat_json_master(target_dir,nnodes_active,vars):
    ''' 
    Reads all json-files created by slave nodes in 'process_data'-function (see above),
    computes final statistics and writes them in final file to be used in subsequent steps.
    '''
 

    all_stat_files = glob.glob(target_dir+"/**/stat_*.json",recursive=True)


    nfiles         = len(all_stat_files)
    if (nfiles < nnodes_active):
       raise ValueError("Found less files than expected by number of active slave nodes!")

  

    vars_uni = np.unique(vars)
    nvars    = len(vars_uni)

    varmin, varmax = np.full(nvars,np.nan), np.full(nvars,np.nan)   # initializes with NaNs -> make use of np.fmin/np.fmax subsequently
    varavg         = np.zeros(nvars)

    for ff in range(nfiles):
        with open(all_stat_files[ff]) as js_file:
            data = json.load(js_file)
            
            varmin, varmax = np.fmin(varmin,get_stat(data,"min")), np.fmax(varmax,get_stat(data,"max"))
            varavg        += get_stat(data,"avg")
            
    # write final statistics
    stat_dict = {}
    for i in range(nvars):
        stat_dict[vars_uni[i]]=[]
        stat_dict[vars_uni[i]].append({
                  'min': varmin[i],
                  'max': varmax[i],
                  'avg': varavg[i]/nfiles

        })

    js_file = os.path.join(target_dir+"/splits",'statistics.json')
    with open(js_file,'w') as stat_out:
        json.dump(stat_dict, stat_out)
    print(js_file+" was created successfully...")
            

# ML 2020/04/03 E

# ML 2020/05/15 S
class calc_data_stat:
    """Class for computing statistics and saving them to a json-files."""
    def __init__(self,nvars):
        self.stat_dict = {}
        self.varmin    = np.full(nvars,np.nan)
        self.varmax    = np.full(nvars,np.nan)
        self.varavg    = np.zeros(nvars)
        self.nfiles    = 0

    def acc_stat(self,ivar,data):
        self.varmin[ivar]  = np.fmin(self.varmin[ivar],np.amin(data))
        self.varmax[ivar]  = np.fmax(self.varmax[ivar],np.amax(data))
        self.varavg[ivar] += np.average(data)
        if (ivar == 0): self.nfiles += 1 
        
    def finalize_stat_loc(self,varnames):
        vars_uni, varsind = np.unique(varnames,return_index=True)
        nvars = len(vars_uni)

        varmin, varmax, varavg = self.varmin[varsind], self.varmax[varsind], self.varavg[varsind] 
        
        for i in range(nvars):
            varavg[i] /= self.nfiles
            print('varavg['+str(i)+'] : {0:5.2f}'.format(varavg[i]))
            print('length of imageList: ',self.nfiles)

            self.stat_dict[vars_uni[i]]=[]
            self.stat_dict[vars_uni[i]].append({
                  'min': varmin[i],
                  'max': varmax[i],
                  'avg': varavg[i]
            })        
        self.stat_dict["common_stat"] = [
            {"nfiles":self.nfiles}]
        
    def write_stat_json_loc(self,path_out,job_name):
        try:
            js_file = os.path.join(path_out,'stat_'+str(job_name) + '.json')
            with open(js_file,'w') as stat_out:
                json.dump(self.stat_dict,stat_out)
        except ValueError: 
            print("Something went wrong when writing dictionary to json-file: '"+js_file+"''")
        finally:
            print("Created statistics json-file '"+js_file+"' successfully.")

# ML 2020/05/15 E
                 

def split_data_multiple_years(target_dir,partition):
    """
    Collect all the X_*.hkl data across years and split them to training, val and testing datatset
    """

    #target_dirs = [os.path.join(target_dir,year) for year in years]
    #os.chdir(target_dir)
    splits_dir = os.path.join(target_dir,"splits")
    os.makedirs(splits_dir, exist_ok=True) 
    splits = {s: [] for s in list(partition.keys())}
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
        X = np.array(X) 
        print("==================={}=====================".format(split))
        print ("Sources for {} dataset are {}".format(split,files))
        print("Number of images in {} dataset is {} ".format(split,len(X)))
        print ("dataset shape is {}".format(np.array(X).shape))
        hkl.dump(X, os.path.join(splits_dir , 'X_' + split + '.hkl'))
        hkl.dump(files, os.path.join(splits_dir,'sources_' + split + '.hkl'))
        

        
    

                
            
                
            
            
        
    
    
    
    
    
    



    

