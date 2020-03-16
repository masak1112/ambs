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
from netCDF4 import Dataset


#TODO: Not optimal with DATA_DIR and filingPath: In original process_kitti.py 
# there's just DATA_DIR (which is specified in kitti_settings.py) and in there
# the processed data will be stores. The raw data lies also in there in a subfolder

#Path of ERA5 Data
#DATA_DIR = '/p/scratch/cjjsc42/bing/pystager-development/tryData/01'
#Path where to save processed data
#target_dir = '/p/scratch/cjjsc42/bing/pystager-development/processedData'

# print('Image List:')
# print(imageList)
# print('Length of Image List: ' + str(len(imageList)))
#scp hussmann1@jureca.fz-juelich.de:/p/project/cjjsc42/severin/data/era5_extract_481.nc ~/Desktop/netCDFdata

# ToDo: Define properly the train, val and test index
# Here just for testing and taking weird .DS_Store file into consideration
# http://www.apfelwiki.de/Main/DSStore
#train_recordings = imageList[1:6]
#val_recordings = imageList[7:9]
#test_recordings = imageList[-2:]

#Train,Val,Test size in percentage
#partition = [0.8, 0.05, 0.15]
#determine correct indices 
#train_begin = 0
#train_end = round(partition[0]*len(imageList))-1
#val_begin = train_end + 1
#val_end = train_end + round(partition[1]*len(imageList))
#test_begin = val_end + 1
#test_end = len(imageList)-1
#print('Indices of Train, Val and test: '+ str(train_begin) + ' ' + str(val_begin) + ' ' + str(test_begin))
#slightly adapting start and end because starts at the first index given and stops before(!) the last. 
#train_recordings = imageList[train_begin:val_begin]
#val_recordings = imageList[val_begin:test_begin]
#test_recordings = imageList[test_begin:test_end]

#adapted for feature testing: just first year (2015); Otherwise would take too long and some weird mistake in some data in 2016
#in total: 17544
#half: 8772
#train: 0-6900
#val:6901-7000
#test:7001-8772
#train_recordings = imageList[0:1000]
#val_recordings = imageList[6901:7000]
#test_recordings = imageList[7001:8772]

# print('Now everything together:')
# print('Train:')
# print(train_recordings)
# print('Val:')
# print(val_recordings)
# print('Test:')
# print(tiest_recordings)
desired_im_sz = (64, 64)
# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data(directory_to_process,target_dir,job_name):
    # ToDo: Define a convenient function to create a list containing all files.
    imageList = list(os.walk(directory_to_process, topdown = False))[-1][-1]
    imageList = sorted(imageList)
    EU_stack_list = [0] * (len(imageList))

    #X = np.zeros((len(splits[split]),) + desired_im_sz + (3,), np.uint8)
    #print(X)
    #print('shape of X' + str(X.shape))

    ##### TODO: iterate over split and read every .nc file, cut out array,
    #####		overlay arrays for RGB like style.
    #####		Save everything after for loop.
    EU_stack_list = [0] * (len(imageList))

    for i, im_file in enumerate(imageList):
        try:
            im_path = os.path.join(directory_to_process, im_file)
            print('Open following dataset: ' + im_path)
            im = Dataset(im_path, mode = 'r')
            #print(im)
            t2 = im.variables['T2'][0,:,:]
            msl = im.variables['MSL'][0,:,:]
            gph500 = im.variables['gph500'][0,:,:]
            im.close()
            EU_t2 = t2[74+32:202-32,(550+16+32):(710-16-32)]
            EU_msl = msl[74+32:202-32, (550+16+32):(710-16-32)]
            EU_gph500 = gph500[74+32:202-32,(550+16+32):(710-16-32)]
            #print(EU_t2.shape, EU_msl.shape, EU_gph500.shape)
            #Normal stack: T2, MSL & GPH500
            #EU_stack = np.stack([EU_t2, EU_msl, EU_gph500],axis=2)
            #Stack T2 only:
            #EU_stack = np.stack([EU_t2, EU_t2, EU_t2],axis=2)
            #EU_stack_list[i]=EU_stack
            #Stack T2*2 MSL*1:
            #EU_stack = np.stack([EU_t2, EU_t2, EU_msl],axis=2)
            #EU_stack_list[i]=EU_stack
            #EU_stack = np.stack([EU_t2, EU_msl, EU_msl],axis=2)
            #EU_stack_list[i]=EU_stack
            #Stack T2*2 gph500*1:
            #EU_stack = np.stack([EU_t2, EU_t2, EU_gph500],axis=2)
            #EU_stack_list[i]=EU_stack
            #Stack T2*1 gph500*2
            #EU_stack = np.stack([EU_t2, EU_gph500, EU_gph500],axis=2)
            #EU_stack_list[i]=EU_stack
            #print(EU_stack.shape)
            #X[i]=EU_stack #this should be unnecessary
            #t2_1 stack. Stack t2 with two empty arrays
            #empty_image = np.zeros(shape = (128, 160))
            #EU_stack = np.stack([EU_t2, empty_image, empty_image],axis=2)
            #EU_stack_list[i]=EU_stack
            #t2_2 stack. Stack t2 with one empty array
#            empty_image = np.zeros(shape = (64, 64))
            EU_stack = np.stack([EU_t2, EU_t2, EU_t2],axis=2)
            EU_stack_list[i] =list(EU_stack)
            #print('Does ist work? ')
            #print(EU_stack_list[i][:,:,0]==EU_t2)
            #print(EU_stack[:,:,1]==EU_msl
        except Exception as err:
            print("*************ERROR*************", err)
            print("Error message {} from file {}".format(err,im_file))
            EU_stack_list[i] = list(EU_stack) # use the previous image as replacement, we can investigate further how to deal with the missing values
            continue
            
    X = np.array(EU_stack_list)
    print('Shape of X: ' + str(X.shape))
    target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.hkl')
    hkl.dump(X, target_file) #Not optimal!
    print(target_file, "is saved")
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

def process_netCDF_in_dir(job_name,src_dir,target_dir):
    print ("job_name",job_name)
    directory_to_process = os.path.join(src_dir, job_name)
    print("Going to process file in directory {}".format(directory_to_process))
    os.chdir(directory_to_process)
    if not os.path.exists(target_dir): os.mkdir(target_dir)
    target_file = os.path.join(target_dir, 'X_' + str(job_name) + '.hkl')
    if os.path.exists(target_file):
        print(target_file," file exists in the directory ", target_dir)
    else:
        print ("==========Processing files in directory {} =============== ".format(directory_to_process))
        process_data(directory_to_process=directory_to_process, target_dir=target_dir, job_name=job_name)


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





