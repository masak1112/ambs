from mpi4py import MPI
from os import walk
import sys
import subprocess
import logging
import time
import shutil
import glob
import argparse
import os


from helper_single_master import directory_scanner
from helper_single_master import load_distributor
from helper_single_master import hash_directory
from helper_single_master import data_structure_builder
from helper_single_master import md5

from prepare_era5_data import prepare_era5_data_one_file

# How to Run it!
# mpirun -np 6 python mpi_stager_v2.py
# mpiexec -np 6 python mpi_stager_v2.py


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job_id",type=int,default=100)
    parser.add_argument("--source_dir",type=str,default="//home/a.mozaffari/data_era5/2017/")
    parser.add_argument("--destination_dir",type=str,default="/home/a.mozaffari/data_dest/")
    parser.add_argument("--log_temp",type=str,default="log_temp")
    parser.add_argument("--checksum_status",type=int,default = 0)
    parser.add_argument("--rsync_status",type=int,default=0)
    parser.add_argument("--load_level",type=int,default=0)
    parser.add_argument("--clear_destination",type=int,default=1)
    args = parser.parse_args()
    # for the local machine test
    current_path = os.getcwd()
    job_id = args.job_id
    source_dir = args.source_dir
    destination_dir = args.destination_dir
    checksum_status = args.checksum_status
    rsync_status = args.rsync_status
    clear_destination = args.clear_destination
    log_temp = args.log_temp


    # for the local machine test
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_path)
    time.sleep(0)

# ini. MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()  # rank of the node
    p = comm.Get_size()  # number of assigned nods
    firs_slave_processor_id = 1


    # ==================================== Master Logging ==================================================== #
    # DEBUG: Detailed information, typically of interest only when diagnosing problems.
    # INFO: Confirmation that things are working as expected.
    # WARNING: An indication that something unexpected happened, or indicative of some problem in the near
    # ERROR: Due to a more serious problem, the software has not been able to perform some function.
    # CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
    # It will copy the logging messages to the stdout, for the case of container version on HPC

    if my_rank == 0:  # node is master

    # delete the general logger if exist
        logger_path = current_path + '/distribution_job_{job_id}.log'.format(job_id=job_id)
        if os.path.isfile(logger_path):
            print("Logger Exists -> Logger Deleted")
            os.remove(logger_path)
        logging.basicConfig(filename='distribution_job_{job_id}.log'.format(job_id=job_id), level=logging.DEBUG,
                            format='%(asctime)s:%(levelname)s:%(message)s')
        logger = logging.getLogger(__file__)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        start = time.time()  # start of the MPI

# check the existence of the  source path :
    if not os.path.exists(source_dir):  # check if the source dir. is existing
        if my_rank == 0:
            logger.critical('The source does not exist')
            message_out = "Source : {source} is not existing -> Abort".format(source=source_dir)
            logger.info('exit status : 1')
        sys.exit(1)

# Check if the destination is existing, if so, it will delete and recreate the destination_dir
    if os.path.exists(destination_dir):
        if my_rank == 0:
            logger.info('The destination exist')
            if clear_destination == 1:
                shutil.rmtree(destination_dir)
                os.mkdir(destination_dir)
                logger.critical("Destination : {destination} exist -> Remove and Re-Cereate".format(destination=destination_dir))
                print("Destination : {destination} exist -> Remove and Re-Cereate".format(destination=destination_dir))

            else:
                logger.critical("Destination : {destination} exist -> will not be removed (caution : overwrite)".format(destination=destination_dir))
                print("Destination : {destination} exist -> will not be rmeoved (caution : overwrite)".format(destination=destination_dir))

    # Create a log folder for slave-nodes to write down their processes
    slave_log_path = os.path.join(destination_dir,log_temp)

    if my_rank == 0:
        if os.path.exists(slave_log_path) == False:
            os.mkdir(slave_log_path)

    if my_rank == 0:  # node is master

    # ==================================== Master : Directory scanner {Parent level load level = 0}  ================================= #

        logger.info("The source path is  : {path}".format(path=source_dir))
        logger.info("The destination path is  : {path}".format(path=destination_dir))
        logger.info("==== Directory scanner : start ====")
        load_level = 0
        ret_dir_scanner = directory_scanner(source_dir,load_level)
    #print(ret_dir_scanner)

    # Unifying the naming of this section for both cases : Sub - Directory or File
    # dir_detail_list == > Including the name of the directories, size and number of teh files in each directory / for files is empty
    # list_items_to_process    === > List of items to process  (Sub-Directories / Files)
    # total_size_source  === > Total size of the items to process
    # total_num_files    === > for Sub - Directories : sum of all files in different directories / for Files is sum of all
    # total_num_directories  === > for Files = 0

        dir_detail_list = ret_dir_scanner[0]
        list_items_to_process = ret_dir_scanner[1]
        total_size_source = ret_dir_scanner[2]
        total_num_files = ret_dir_scanner[3]
        total_num_dir = ret_dir_scanner[4]
        logger.info("==== Directory scanner : end ====")

    # ================================= Master : Data Structure Builder {Parent level load level = 0} ========================= #

        logger.info("==== Data Structure Builder : start  ====")
        data_structure_builder(source_dir, destination_dir, dir_detail_list, list_items_to_process,load_level)
        logger.info("==== Data Structure Builder : end  ====")
        # message to inform the slaves that they will recive #Batch of messages including the logger_p
        batch_info = list_items_to_process
        for slaves in range (1,p):
            comm.send(batch_info, dest=slaves)

        for batch_counter in range (0,len(batch_info)):
            #relative_source =  source_dir + str(batch_info[batch_counter]) +"/"
            relative_source = os.path.join(source_dir,str(batch_info[batch_counter]))
            print(relative_source)
            logger.info("MA{my_rank}: Next to be processed is {task} loacted in  {path} ".format(my_rank = my_rank,task=batch_info[batch_counter], path=relative_source))
            load_level = 1 # it will process the files in the relative source

        #________ Directory Scanner ______#
            relative_ret_dir_scanner = directory_scanner(relative_source,load_level)
            relative_dir_detail_list = relative_ret_dir_scanner[0]
            relative_list_items_to_process = relative_ret_dir_scanner[1]
            relative_total_size_source = relative_ret_dir_scanner[2]
            relative_total_num_files = relative_ret_dir_scanner[3]
            relative_total_num_dir = relative_ret_dir_scanner[4]
        #________ Load Distribution ________#
            relative_ret_load_balancer = load_distributor(relative_dir_detail_list, relative_list_items_to_process, relative_total_size_source, relative_total_num_files, relative_total_num_dir,load_level, p)
            relative_transfer_dict = relative_ret_load_balancer
            logger.info(relative_transfer_dict)

        #________ Communication ________#

            for processor in range(firs_slave_processor_id, p):
                broadcast_list = relative_transfer_dict[processor]
                comm.send(broadcast_list, dest=processor)

        receive_counter = 0
        total_number_messages = (p-1) * len(batch_info) - 1
        while receive_counter <= total_number_messages:
            message_in = comm.recv()
            logger.info("MA{my_rank}: S{message_in} ".format(my_rank=my_rank,message_in=message_in))
            receive_counter = receive_counter + 1


        # Cleaning up the slaves temprory log file, if it is empty.
        if len(os.listdir(slave_log_path) ) == 0:
            print("Temprory log file is empty, it is deleted")
            os.removedirs(slave_log_path)


        end = time.time()
        termination_message = "MA{my_rank}: Sucssfully terminated with total time : {wall_time}".format(my_rank=my_rank,wall_time= end-start)
        logger.info(termination_message)
        sys.exit(0)

    else:  # Processor is slave

    # ============================================= Slave : Send / Receive ============================================ #
    # recive the #Batch process that will be recived
        batch_info = comm.recv(source = 0)
        #print("S{my_rank} will receive {todo_message} batch of task to process".format(my_rank=my_rank, todo_message=len(batch_info)))
        batch_counter = 0

    # here will be a loop around all the #batchs

        while batch_counter <= len(batch_info) -1:
            message_in = comm.recv(source = 0)
            relative_source_directory = os.path.join(source_dir,str(batch_info[batch_counter]))
            relative_destination_directory = os.path.join(destination_dir,str(batch_info[batch_counter]))

            if message_in is None:  # in case more than number of the dir. processor is assigned !
                slave_out_message = "{my_rank} is idle".format(my_rank=my_rank)
                # comm.send(message_out, dest=1)

            else: # if the Slave node has joblist to do
                job_list = message_in.split(';')
                for job_count in range(0, len(job_list)):
                    job = job_list[job_count] # job is the name of the directory(ies) assigned to slave_node
                    #print(job)
                    if rsync_status == 1:
                        # prepare the rsync commoand to be excexuted by the worker node
                        rsync_message = "rsync {relative_source_directory}/{job} {relative_destination_directory}/{job}".format(relative_source_directory=relative_source_directory,job=job, relative_destination_directory=relative_destination_directory)
                        os.system(rsync_message)
                        #slave_out_message= " RSYNC process"
                    else :
                        ## @Bing here is the job for the slaves
                        print("S{my_rank} will execute era5 preperation on {job}".format(my_rank=my_rank, job=job))
                        prepare_era5_data_one_file(src_file=job,directory_to_process=relative_source_directory, target=job, target_dir=relative_destination_directory)



                        #if job.endswith(".nc"):
                        #    if os.path.exists(os.path.join(relative_destination_directory, job)):
                        #        print("{job} is has been processed in directory {directory}".format(job=job,directory=relative_destination_directory))
                        #else:
                        #    prepare_era5_data_one_file(src_file=job,directory_to_process=relative_source_directory, target=job, target_dir=relative_destination_directory)
                        #    print("File {job} in directory {directory} has been processed in directory".format(job=job,directory=relative_destination_directory))
                        #
                        #slave_out_message = " {in_message} process".format(in_message=my_rank)
                        # Generate a hash of the output

            message_out = "{my_rank}: is finished the {in_message} .".format(my_rank=my_rank,in_message=batch_info[batch_counter])
            comm.send(message_out, dest=0)
            batch_counter = batch_counter + 1

    MPI.Finalize()


if __name__ == "__main__":
    main()
