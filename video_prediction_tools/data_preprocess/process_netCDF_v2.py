'''
Code for processing staged ERA5 data 
'''

import os
import glob
from netCDF4 import Dataset,num2date
from statistics import Calc_data_stat
#import requests
#from bs4 import BeautifulSoup
#import urllib.request
import numpy as np
#from imageio import imread
#from scipy.misc import imresize
import json
import pickle

# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data(directory_to_process, target_dir, job_name, slices, vars=("T2","MSL","gph500")):
    '''
    :param directory_to_process: directory where netCDF-files are stored to be processed
    :param target_dir: directory where pickle-files will e stored
    :param job_name: job_id passed and organized by PyStager
    :param slices: indices defining geographical region of interest
    :param vars: variables to be processed
    :return: Saves pickle-files which contain the sliced meteorological data and temporal information as well
    '''
    desired_im_sz = (slices["lat_e"] - slices["lat_s"], slices["lon_e"] - slices["lon_s"])
    # ToDo: Define a convenient function to create a list containing all files.
    imageList = list(os.walk(directory_to_process, topdown = False))[-1][-1]
    imageList = sorted(imageList)
    EU_stack_list = [0] * (len(imageList))
    temporal_list = [0] * (len(imageList))
    nvars = len(vars)

    # ML 2020/04/06 S
    # Some inits
    stat_obj = Calc_data_stat(nvars)
    # ML 2020/04/06 E
    for j, im_file in enumerate(imageList):
        try:
            im_path = os.path.join(directory_to_process, im_file)
            print('Open following dataset: '+im_path)
            vars_list = []
            with Dataset(im_path,'r') as data_file:
                times = data_file.variables['time']
                time = num2date(times[:],units=times.units,calendar=times.calendar)
                for i in range(nvars):
                    var1 = data_file.variables[vars[i]][0,slices["lat_s"]:slices["lat_e"], slices["lon_s"]:slices["lon_e"]]
                    stat_obj.acc_stat_loc(i,var1)
                    vars_list.append(var1)

            EU_stack = np.stack(vars_list, axis = 2)
            EU_stack_list[j] =list(EU_stack)

            #20200408,bing
            temporal_list[j] = list(time)
        except Exception as err:
            im_path = os.path.join(directory_to_process, im_file)
            #im = Dataset(im_path, mode = 'r')
            print("*************ERROR*************", err)
            print("Error message {} from file {}".format(err,im_file))
            EU_stack_list[j] = list(EU_stack) # use the previous image as replacement, we can investigate further how to deal with the missing values
            continue
            
    X = np.array(EU_stack_list)
    # ML 2020/07/15: Make use of pickle-files only
    target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.pkl')
    with open(target_file, "wb") as data_file:
        pickle.dump(X,data_file)
    #target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.pkl')    
    #hkl.dump(X, target_file) #Not optimal!
    print(target_file, "is saved")
    # ML 2020/03/31: write json file with statistics
    stat_obj.finalize_stat_loc(vars)
    stat_obj.write_stat_json(target_dir,file_id=job_name)
    # BG 2020/04/08: Also save temporal information to pickle-files
    temporal_info = np.array(temporal_list)
    temporal_file = os.path.join(target_dir, 'T_' + str(job_name) + '.pkl')
    cwd = os.getcwd()
    with open(temporal_file,"wb") as ftemp:
        pickle.dump(temporal_info,ftemp)
    #pickle.dump(temporal_info, open( temporal_file, "wb" ) )

def process_netCDF_in_dir(src_dir,**kwargs):
    target_dir = kwargs.get("target_dir")
    job_name = kwargs.get("job_name")
    directory_to_process = os.path.join(src_dir, job_name)
    os.chdir(directory_to_process)
    if not os.path.exists(target_dir): os.mkdir(target_dir)
    #target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.hkl')
    # ML 2020/07/15: Make use of pickle-files only
    target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.hkl')
    if os.path.exists(target_file):
        print(target_file," file exists in the directory ", target_dir)
    else:
        print ("==========Processing files in directory {} =============== ".format(directory_to_process))
        process_data(directory_to_process=directory_to_process, **kwargs)


# ML 2020/08/03 Not used anymore!
#def split_data_multiple_years(target_dir,partition,varnames):
    #"""
    #Collect all the X_*.hkl data across years and split them to training, val and testing datatset
    #"""
    ##target_dirs = [os.path.join(target_dir,year) for year in years]
    ##os.chdir(target_dir)
    #splits_dir = os.path.join(target_dir,"splits")
    #os.makedirs(splits_dir, exist_ok=True) 
    #splits = {s: [] for s in list(partition.keys())}
    ## ML 2020/05/19 S
    #vars_uni, varsind, nvars = get_unique_vars(varnames)
    #stat_obj = Calc_data_stat(nvars)
    
    #for split in partition.keys():
        #values = partition[split]
        #files = []
        #X = []
        #Temporal_X = []
        #for year in values.keys():
            #file_dir = os.path.join(target_dir,year)
            #for month in values[year]:
                #month = "{0:0=2d}".format(month)
                #hickle_file = "X_{}.hkl".format(month)
                ##20200408:bing
                #temporal_file = "T_{}.pkl".format(month)
                ##data_file = os.path.join(file_dir,hickle_file)
                #data_file = os.path.join(file_dir,hickle_file)
                #temporal_data_file = os.path.join(file_dir,temporal_file)
                #files.append(data_file)
                #data = hkl.load(data_file)
                #with open(temporal_data_file,"rb") as ftemp:
                    #temporal_data = pickle.load(ftemp)
                #X = X + list(data)
                #Temporal_X = Temporal_X + list(temporal_data)
                ## process stat-file:
                #stat_obj.acc_stat_master(file_dir,int(month))
        #X = np.array(X) 
        #Temporal_X = np.array(Temporal_X)
        #print("==================={}=====================".format(split))
        #print ("Sources for {} dataset are {}".format(split,files))
        #print("Number of images in {} dataset is {} ".format(split,len(X)))
        #print ("dataset shape is {}".format(np.array(X).shape))
        ## ML 2020/07/15: Make use of pickle-files only
        #with open(os.path.join(splits_dir , 'X_' + split + '.pkl'),"wb") as data_file:
            #pickle.dump(X,data_file,protocol=4)
        ##hkl.dump(X, os.path.join(splits_dir , 'X_' + split + '.hkl'))

        #with open(os.path.join(splits_dir,"T_"+split + ".pkl"),"wb") as temp_file:
            #pickle.dump(Temporal_X, temp_file)
        
        #hkl.dump(files, os.path.join(splits_dir,'sources_' + split + '.hkl'))
        
    ## write final statistics json-file
    #stat_obj.finalize_stat_master(target_dir,vars_uni)
    #stat_obj.write_stat_json(splits_dir)

        
    


                
            
                
            
            
        
    
    
    
    
    
    



    

