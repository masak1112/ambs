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
import hickle as hkl
from netCDF4 import Dataset,num2date
import pickle



# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data(directory_to_process, target_dir, job_name, slices, vars=("T2","MSL","gph500")):
    desired_im_sz = (slices["lat_e"] - slices["lat_s"], slices["lon_e"] - slices["lon_s"])
    # ToDo: Define a convenient function to create a list containing all files.
    imageList = list(os.walk(directory_to_process, topdown = False))[-1][-1]
    imageList = sorted(imageList)
    EU_stack_list = [0] * (len(imageList))
    len_vars = len(vars)
    #X = np.zeros((len(splits[split]),) + desired_im_sz + (3,), np.uint8)
    #print(X)
    #print('shape of X' + str(X.shape))

    ##### TODO: iterate over split and read every .nc file, cut out array,
    #####		overlay arrays for RGB like style.
    #####		Save everything after for loop.
    EU_stack_list = [0] * (len(imageList))
    #20200408,Bing   
    temporal_list = [0] * (len(imageList))
    for i, im_file in enumerate(imageList):
        try:
            im_path = os.path.join(directory_to_process, im_file)
            print('Open following dataset: '+im_path)
             
            
            #20200408,Bing
            
            im = Dataset(im_path, mode = 'r')
            times = im.variables['time']
            time = num2date(times[:],units=times.units,calendar=times.calendar)
            vars_list = []
            for j in range(len_vars):
                var1 = im.variables[vars[j]][0, :, :]
                var1 = var1[slices["lat_s"]:slices["lat_e"], slices["lon_s"]:slices["lon_e"]]
                vars_list.append(var1)
            im.close()

            EU_stack = np.stack(vars_list, axis = 2)

            EU_stack_list[i] =list(EU_stack)
            #20200408,bing
            temporal_list[i] = list(time)
            #print('Does ist work? ')
            #print(EU_stack_list[i][:,:,0]==EU_t2)
            #print(EU_stack[:,:,1]==EU_msl
        except Exception as err:
            im_path = os.path.join(directory_to_process, im_file)
            #im = Dataset(im_path, mode = 'r')
            print("*************ERROR*************", err)
            print("Error message {} from file {}".format(err,im_file))
            EU_stack_list[i] = list(EU_stack) # use the previous image as replacement, we can investigate further how to deal with the missing values
            continue
            
    X = np.array(EU_stack_list)
    target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.hkl')
    hkl.dump(X, target_file) #Not optimal!
    print(target_file, "is saved")
    #20200408:bing
    temporal_info = np.array(temporal_list)
    temporal_file = os.path.join(target_dir, 'T_' + str(job_name) + '.pkl')
    cwd = os.getcwd()
    pickle.dump(temporal_info, open( temporal_file, "wb" ) )
    #hkl.dump(temporal_info, temporal_file) 

        #hkl.dump(source_list, os.path.join(target_dir, 'sources_' + str(job) + '.hkl'))

        #for category, folder in splits[split]:
        #    im_dir = os.path.join(DATA_DIR, 'raw/', category, folder, folder[:10], folder, 'image_03/data/')
        #    files = list(os.walk(im_dir, topdown=False))[-1][-1]
        #    im_list += [im_dir + f for f in sorted(files)]
            # multiply path of respective recording with lengths of its files in order to ensure
            # that each entry in X_train.hkl corresponds with an entry of source_list/ sources_train.hkl
        #    source_list += [category + '-' + folder] * len(files)

        #print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        #X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        # enumerate allows us to loop over something and have an automatic counter
        #for i, im_file in enumerate(im_list):
        #    im = imread(im_file)
        #    X[i] = process_im(im, desired_im_sz)

        #hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        #hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))

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
    #Bing: 20200415
    temporal_files = glob.glob("*.pkl")
    temporal_filesList = sorted(temporal_files)

    # determine correct indicesue
    train_begin = 0
    train_end = round(partition[0] * len(filesList)) - 1
    val_begin = train_end + 1
    val_end = train_end + round(partition[1] * len(filesList))
    test_begin = val_end + 1
   
    
    # slightly adapting start and end because starts at the first index given and stops before(!) the last.
    train_files = filesList[train_begin:val_begin]
    val_files = filesList[val_begin:test_begin]
    test_files = filesList[test_begin:]
    #bing: 20200415
    train_temporal_files = temporal_filesList[train_begin:val_begin]
    val_temporal_files = temporal_filesList[val_begin:test_begin]
    test_temporal_files = temporal_filesList[test_begin:]


    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_files
    splits['test'] = test_files
    splits['train'] = train_files


    splits_temporal = {s: [] for s in ['train', 'test', 'val']}
    splits_temporal["train"] = train_temporal_files
    splits_temporal["val"] = val_temporal_files
    splits_temporal["test"] = test_temporal_files
    
    for split in splits:
        X = []
        X_temporal = []
        files = splits[split]
        temporal_files = splits_temporal[split]
        for file, temporal_file in zip(files, temporal_files):
            data_file = os.path.join(target_dir,file)
            temporal_file = os.path.join(target_dir,temporal_file)
            #load data with hkl file
            data = hkl.load(data_file)
            temporal_data = pickle.load(open(temporal_file,"rb"))
            X_temporal = X_temporal + list(temporal_data)
            X = X + list(data)
        X = np.array(X)
        X_temporal = np.array(X_temporal)
        print ("X_temporal",X_temporal)
        #save training, val and test data into splits directoyr
        hkl.dump(X, os.path.join(split_dir, 'X_' + split + '.hkl'))
        hkl.dump(files, os.path.join(split_dir,'sources_' + split + '.hkl'))
        pickle.dump(X_temporal,open(os.path.join(split_dir,"T_"+split + ".pkl"),"wb"))
        print ("PICKLE FILE FOR SPLITS SAVED")
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
                #20200408:bing
                temporal_file = "T_{}.hkl".format(month)
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
        

        
    


                
            
                
            
            
        
    
    
    
    
    
    



    

