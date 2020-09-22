from mpi4py import MPI
from os import walk
import sys
import subprocess
import logging
import time
from utils.external_function import directory_scanner
from utils.external_function import load_distributor
from utils.external_function import hash_directory
from utils.external_function import md5
from utils.prepare_era5_data import *
# How to Run it!
# mpirun -np 6 python mpi_stager_v2.py
import os
from pathlib import Path
import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source_dir",type=str,default="//home/a.mozaffari/data_era5/2017/")
    parser.add_argument("--destination_dir",type=str,default="/home/a.mozaffari/data_dest")
    parser.add_argument("--checksum_status",type=int,default = 0)
    parser.add_argument("--rsync_status",type=int,default=1)
    args = parser.parse_args()
    # for the local machine test
    current_path = os.getcwd()
    source_dir = args.source_dir
    destination_dir = args.destination_dir
    checksum_status = args.checksum_status
    rsync_status = args.rsync_status


    # for the local machine test
    current_path = os.getcwd()
    # TODO : it will be integerated in the seperated read_in_file
    #rot_grid="/mnt/rasdaman/DeepRain/gridneu.dat"
    os.chdir(current_path)
    time.sleep(0)
    # ini. MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()  # rank of the node
    p = comm.Get_size()  # number of assigned nods

    # ============ configuration for data preprocessing =================== #

    # ==================================== Master Logging ==================================================== #
    # DEBUG: Detailed information, typically of interest only when diagnosing problems.
    # INFO: Confirmation that things are working as expected.
    # WARNING: An indication that something unexpected happened, or indicative of some problem in the near
    # ERROR: Due to a more serious problem, the software has not been able to perform some function.
    # CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

    if my_rank == 0:  # node is master
        logging.basicConfig(filename='stager.log', level=logging.DEBUG,
                            format='%(asctime)s:%(levelname)s:%(message)s')
        start = time.time()  # start of the MPI
        logging.debug(' === PyStager is started === ')
        print('PyStager is Running .... ')

    # ================================== ALL Nodes:  Read-in parameters ====================================== #

    # fileName = "parameters.dat"  # input parameters file
    # fileObj = open(fileName)
    # params = {}

    # for line in fileObj:
    #     line = line.strip()
    #     read_in_value = line.split("=")
    #     if len(read_in_value) == 2:
    #         params[read_in_value[0].strip()] = read_in_value[1].strip()

    # # input from the user:
    # source_dir = str(params["Source_Directory"])
    # destination_dir = str(params["Destination_Directory"])
    # log_dir = str(params["Log_Directory"])
    # rsync_status = int(params["Rsync_Status"])
    # checksum_status = int(params["Checksum_Status"])



    # check the existence of teh folders :

    if not os.path.exists(source_dir):  # check if the source dir. is existing
        if my_rank == 0:
            logging.critical('The source does not exist')
            logging.info('exit status : 1')
            print('Critical : The source does not exist')

        sys.exit(1)

    if not os.path.exists(destination_dir):  # check if the Destination dir. is existing
        if my_rank == 0:
            logging.critical('The Destination does not exist')
            logging.info('Create a Destination dir')
            if not os.path.exists(destination_dir): os.makedirs(destination_dir)
            print('Create a Destination dir')



    if my_rank == 0:  # node is master

        # ==================================== Master : Directory scanner ================================= #

        print(" # ==============  Directory scanner : start    ==================# ")

        ret_dir_scanner = directory_scanner(source_dir)
        print(ret_dir_scanner)

        dir_detail_list = ret_dir_scanner[0]
        sub_dir_list = ret_dir_scanner[1]
        total_size_source = ret_dir_scanner[2]
        total_num_files = ret_dir_scanner[3]
        total_num_dir = ret_dir_scanner[4]

        # ===================================  Master : Load Distribution   ========================== #

        print(" # ==============  Load Distrbution  : start  ==================# ")
        #def load_distributor(dir_detail_list, sub_dir_list, total_size_source, total_num_files, total_num_directories, p):
        ret_load_balancer = load_distributor(dir_detail_list, sub_dir_list, total_size_source, total_num_files, total_num_dir, p)
        transfer_dict = ret_load_balancer


        print(ret_load_balancer)

        # ===================================== Master : Send / Receive =============================== #
        print(" # ==============  Communication  : start  ==================# ")

        # Send : the list of the directories to the nodes
        for nodes in range(1, p):
            broadcast_list = transfer_dict[nodes]
            comm.send(broadcast_list, dest=nodes)

        # Receive : will wait for a certain time to see if it will receive any critical error from the slaves nodes
        idle_counter = p - len(sub_dir_list)
        while idle_counter > 1:  # non-blocking receive function
            message_in = comm.recv()
            logging.warning(message_in)
            #print('Warning:', message_in)
            idle_counter = idle_counter - 1

        # Receive : Message from slave nodes confirming the sync
        message_counter = 1
        while message_counter <= len(sub_dir_list):  # non-blocking receive function
            message_in = comm.recv()
            logging.info(message_in)
            message_counter = message_counter + 1

        # stamp the end of the runtime
        end = time.time()
        logging.debug(end - start)
        logging.info('== PyStager is done ==')
        logging.info('exit status : 0')
        print('PyStager is finished ')


        sys.exit(0)

    else:  # node is slave

        # ============================================= Slave : Send / Receive ============================================ #
        message_in = comm.recv()

        if message_in is None:  # in case more than number of the dir. processor is assigned todo Tag it!
            message_out = ('Node', str(my_rank), 'is idle')
            comm.send(message_out, dest=0)

        else: # if the Slave node has joblist to do
            job_list = message_in.split(';')

            for job_count in range(0, len(job_list)):
                job = job_list[job_count] # job is the name of the directory(ies) assigned to slave_node
                #print(job)

                #grib_2_netcdf(rot_grid,source_dir, destination_dir, job)

                # creat a checksum ( hash) from the source folder.
                if checksum_status == 1:
                    hash_directory(source_dir, job, current_path,"source")

                if rsync_status == 1:
                    # prepare the rsync commoand to be excexuted by the worker node
                    rsync_str = ("rsync -r " + source_dir + job + "/" + " " + destination_dir + "/" + job)
                    #os.system(rsync_str)

                    process_era5_in_dir(job, src_dir=source_dir, target_dir=destination_dir)

                    if checksum_status == 1:
                        hash_directory(destination_dir,job,current_path,"destination")
                        os.chdir(current_path)
                        source_hash_text = "source" + "_"+ job +"_hashed.txt"
                        destination_hash_text = "destination"  + "_"+ job +"_hashed.txt"
                        if md5(source_hash_text) == md5(destination_hash_text):
                            msg_out = 'source: ' + job +' and destination: ' + job +' files are identical'
                            print(msg_out)

                        else:
                            msg_out = 'integrity of source: ' + job +' and destination: ' + job +' files could not be verified'
                            print(msg_out)

                else:
                    rsync_str = "None"

                # Send : the finish of the sync message back to master node

                message_out = ('Node:', str(my_rank), 'finished :', rsync_str, '\r\n')
                comm.send(message_out, dest=0)

    MPI.Finalize()



if __name__ == "__main__":
    main()
