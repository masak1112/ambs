#from mpi4py import MPI
from os import walk
import os
import sys
import subprocess
import logging
import time

 
# ======================= List of functions ====================================== #
 
 
def directory_scanner(source_path):
    # Take a look inside a directories and make a list of ll the folders, sub directories, number of the files and size
    # NOTE : It will neglect if there is a sub-directories inside directories!!!
 
    dir_detail_list = []  # directories details
    sub_dir_list = []
    total_size_source = 0
    total_num_files = 0
    list_directories = []
 
    list_directories = os.listdir(source_path)
    print(list_directories)
    print(int(len(list_directories)))
 
    for d in list_directories:
        print(d)
        path = source_path + d
        print(path)
        if os.path.isdir(path):
            sub_dir_list.append(d)
            sub_dir_list.sort()
            num_files = 0
            # size of the files and subdirectories
            size_dir = subprocess.check_output(['du', '-sc', path])
            splitted = size_dir.split()  # fist item is the size of the folder
            size = (splitted[0])
            num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            dir_detail_list.extend([d, size, num_files])
            total_num_files = total_num_files + int(num_files)
            total_size_source = total_size_source + int(size)
        else:
            print(path, 'does not exist')
    print("===== Debug here =====")  
 
    total_num_directories = int(len(list_directories))
    total_size_source = float(total_size_source / 1000000)
 
    message = 'Total size of the source directory is:' + str(total_size_source) + 'Gb.'
    print(message)
    message = "Total number of the files in the source directory is: " + str(total_num_files)
    print(message)
    message = "Total number of the directories  in the source directory is: " + str(total_num_directories)
    print(message)
 
    return dir_detail_list, sub_dir_list, total_size_source, total_num_files, total_num_directories
 
 
def load_distributor(dir_detail_list, sub_dir_list, total_size_source, total_num_files, total_num_directories, p):
    # create a dictionary with p number of keys
    # for each directory they add the name to one of the keys
    print ("range 1 to p is",list(range(1,p)))
    transfer_dict = dict.fromkeys(list(range(1, p)))
    print("transfer_dict:",transfer_dict)
    # package_counter = 0 possibility to use the counter to fill
    counter = 1
    for Directory_counter in range(0, total_num_directories):
 
        if transfer_dict[counter] is None:  # if the value for the key is None add to it
            transfer_dict[counter] = sub_dir_list[Directory_counter]
        else:  # if key has a value join the new value to the old value
            transfer_dict[counter] = "{};{}".format(transfer_dict[counter], sub_dir_list[Directory_counter])
        counter = counter + 1
        if counter == p:
            counter = 1
 
    return transfer_dict
 
def sync_file(source_path, destination_dir, job_name, rsync_status):
    rsync_msg = ("rsync -r " + source_path + job_name + "/" + " " + destination_dir + "/" + job_name)
    # print('Node:', str(my_rank),'will execute :', rsync_str,'\r\n')
    # sync the assigned folder
 
    if rsync_status == 1:
        os.system(rsync_msg)
 
 
