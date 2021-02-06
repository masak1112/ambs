__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Amirpasha Mozaffari"
__date__ = "2020-11-10"




from mpi4py import MPI
import sys
import subprocess
import logging
import time
from utils.external_function import directory_scanner
from utils.external_function import load_distributor
from data_preprocess.prepare_era5_data import *
# How to Run it!
# mpirun -np 6 python mpi_stager_v2.py
import os
import shutil 
from pathlib import Path
import argparse

def main():
    current_path = os.getcwd()

    parser=argparse.ArgumentParser()
    parser.add_argument("--source_dir",type=str,default="//home/a.mozaffari/data_era5/2017/")
    parser.add_argument("--destination_dir",type=str,default="/home/a.mozaffari/data_dest")
    parser.add_argument("--logs_path",type=str,default=current_path)
    args = parser.parse_args()
    # for the local machine test
    current_path = os.getcwd()
    source_dir = args.source_dir
    destination_dir = args.destination_dir
    logs_path = args.logs_path

    os.chdir(current_path)
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
        logs_path = logs_path + '/logs/'
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        
        logger_path_main = logs_path + 'Main_log.log'
        if os.path.exists(logger_path_main):
            print("Logger Exists -> Logger Deleted")
            os.remove(logger_path_main)

        logging.basicConfig(filename=logger_path_main, level=logging.DEBUG,
                            format='%(asctime)s:%(levelname)s:%(message)s')
        logger = logging.getLogger(__file__)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        #start = time.time()  # start of the MPI
        logger.info(' === PyStager is started === ')

    # ================================== ALL Nodes:  Read-in parameters ====================================== #

    # check the existence of teh folders :

    if not os.path.exists(source_dir):  # check if the source dir. is existing
        if my_rank == 0:
            logger.critical('The source does not exist')
            logger.info('exit status : 1')

        sys.exit(1)

    if not os.path.exists(destination_dir):  # check if the Destination dir. is existing
        if my_rank == 0:
            logger.critical('The Destination does not exist')
            logger.info('Create a Destination dir')
            if not os.path.exists(destination_dir): os.makedirs(destination_dir)

    if os.path.exists(destination_dir):
        if my_rank == 0:
            shutil.rmtree(destination_dir)
            os.mkdir(destination_dir)
            logger.critical('The destination exist -> Remove and Re-Create')


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

        # ===================================== Main : Send / Receive =============================== #
        print(" # ==============  Communication  : start  ==================# ")

        # Send : the list of the directories to the nodes
        for nodes in range(1, p):
            broadcast_list = transfer_dict[nodes]
            comm.send(broadcast_list, dest=nodes)

        # All Receive 
        message_counter = 1
        while message_counter < p:  # non-blocking receive function
            message_in = comm.recv()
            Worker_status = message_in[0:5]
            worker_number = message_in[5:7]
            # Idle check Worker_status
            if Worker_status == "IDLEE":
                status = ' An Idle worker is detected, worker number is: {worker_number}'.format(worker_number=worker_number)
                logger.info(status)
            # Success  
            elif Worker_status == "PASSS":
               status =' A job process is finished by worker: {worker_number}'.format(worker_number=worker_number)
               logger.info(status)  

            # Non-Fatal Error 
            elif Worker_status == "NEROR":
               status =' A non-fatal error is triggered by worker: {worker_number}'.format(worker_number=worker_number)
               logger.warning(status)
               logger.warning("System will continue")

            # Fatal Error 
            elif Worker_status == "FEROR":
               status =' A fatal error is triggered by worker: {worker_number}'.format(worker_number=worker_number)
               logger.critical(status)
               logger.critical("System is going to terminate")
               sys.exit(1)
            
            # System fail to recogonise the meesage
            else:
               status =' A message from {worker_number} is not readable by main'.format(worker_number=worker_number)
               logger.critical(status)
               logger.critical("System is going to terminate")
               sys.exit(1)

            message_counter = message_counter + 1

        logger.info(' Main is finished the job and it will terminate the task')
        sys.exit(0)

    else:  # node is slave

        # ============================================= worker: Send / Receive ============================================ #
        # communication works as a break to stop worker before master is ready
        message_in = comm.recv()

        # worker logger file
        worker_log = logs_path + '/logs/' + 'Worker_log_{my_rank}.log'.format(my_rank=my_rank)
        if os.path.exists(worker_log):
            os.remove(worker_log)

        logging.basicConfig(filename=worker_log, level=logging.DEBUG,
                         format='%(asctime)s:%(levelname)s:%(message)s')
        logger = logging.getLogger(__file__)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info('Woker logger is activated')

        # Receive message 
        if message_in is None:  # in case more than number of the dir. processor is assigned todo Tag it!
            message_out = ("IDLEE{worker_rank}: is IDLE ".format(worker_rank=my_rank))
            logger.info('Worker {worker_rank} is idle'.format(worker_rank=my_rank))
            logger.info('Worker {worker_rank} is terminated'.format(worker_rank=my_rank))
            comm.send(message_out, dest=0)
            sys.exit(0)

        else: # if the Worker node has joblist to do
            job_list = message_in.split(';')
            logger.info('Worker {worker_rank} to do list is : {to_do_list}'.format(worker_rank=my_rank,to_do_list=job_list))

            for job_count in range(0, len(job_list)):
                job = job_list[job_count] # job is the name of the directory(ies) assigned to worker
                logger.info('Worker {worker_rank} next job to do is : {job}'.format(worker_rank=my_rank,job=job))

                logger.debug('Worker {worker_rank} is starting the ERA5-preproc. on dir.: {job}'.format(worker_rank=my_rank,job=job))
                
                worker_status = process_era5_in_dir(job, src_dir=source_dir, target_dir=destination_dir)
                
                logger.debug('worker status is: {worker_status}'.format(worker_status=worker_status))
                
                if worker_status == -1:
                    message_out = ("FEROR{worker_rank}:Failed is triggered ".format(worker_rank=my_rank))
                    logger.critical('progress is unsuccessful. fatal-error is observed. Worker is terminating and communicating the termination of the job to main.')
                    comm.send(message_out, dest=0)
                    sys.exit(1)

                if worker_status == 0:
                    logger.debug('progress is successful')
                    message_out = ("PASSS{worker_rank}:is finished".format(worker_rank=my_rank))
                    logger.info('Worker {worker_rank} finished a task'.format(worker_rank=my_rank))

                if worker_status == +1:
                    logger.debug('progress is not successful, but not-fatal')
                    message_out = ("NEROR{worker_rank}:Failed is triggered ".format(worker_rank=my_rank))
                    logger.warning('Worker {worker_rank} has non-fatal failure,but it is continued'.format(worker_rank=my_rank))    

                comm.send(message_out, dest=0)
                sys.exit(0)

    MPI.Finalize()

if __name__ == "__main__":
    main()
