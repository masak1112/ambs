"""
Driver for preprocessing step 1 which parses the input arguments from the runscript
and performs parallelization with PyStager.
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"

from mpi4py import MPI
import os, sys, glob
import logging
import time
import argparse
from utils.external_function import directory_scanner
from utils.external_function import load_distributor
from data_preprocess.process_netCDF_v2 import *  
from metadata import MetaData
from netcdf_datahandling import GeoSubdomain
import json


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", "-src_dir", dest="source_dir", type=str,
                        help="Directory where input netCDF-files are located.")
    parser.add_argument("--destination_dir", "-dest_dir", dest="destination_dir", type=str,
                        help="Destination directory where pickle-files be saved. Note that the complete path is auto-" 
                             "completed during runtime.")
    parser.add_argument("--years", "-y", dest="years", help="Year of data to be processed.")
    parser.add_argument("--rsync_status", type=int, default=1)
    parser.add_argument("--vars", nargs="+", default=["2t", "2t", "2t"], help="Variables to be processed.")
    parser.add_argument("--sw_corner", "-swc", dest="sw_corner", nargs="+", help="Defines south-west corner of target domain " +
                        "(lat, lon)=(-90..90, 0..360)")
    parser.add_argument("--nyx", "-nyx", dest="nyx", nargs="+", help="Number of grid points in zonal and meridional direction.")
    parser.add_argument("--experimental_id", "-exp_id", dest="exp_id", type=str, default="dummy",
                        help="Experimental identifier helping to distinguish between different experiments.")
    args = parser.parse_args()

    current_path = os.getcwd()
    years = args.years
    source_dir = args.source_dir
    source_dir_full = os.path.join(source_dir, str(years))+"/"
    destination_dir = args.destination_dir
    rsync_status = args.rsync_status
   
    vars1 = args.vars
    sw_c = [float(f) for f in args.sw_corner]
    nyx = [int(i) for i in args.nyx]
    print("Selected variables", vars1)

    exp_id = args.exp_id

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

    # check the existence of teh folders :
    if not os.path.exists(source_dir_full):  # check if the source dir. is existing
        if my_rank == 0:
            logging.critical('The source does not exist')
            logging.info('exit status : 1')
            print('Critical : The source does not exist')

        sys.exit(1)
        
    # Expand destination_dir-variable by searching for netCDF-files in source_dir
    # and processing the file from the first list element to obtain all relevant (meta-)data.
    data_files_list = glob.iglob(source_dir_full+"/**/*.nc", recursive=True)
    try:
        data_file = next(data_files_list)
    except StopIteration:
        raise FileNotFoundError("Could not find any data to be processed in '{0}'".format(source_dir_full))

    tar_dom = GeoSubdomain(sw_c, nyx, data_file)

    if my_rank == 0:
        md = MetaData(suffix_indir=destination_dir, exp_id=exp_id, data_filename=data_file, tar_dom=tar_dom,
                      variables=vars1)

        if md.status == "old":          # meta-data file already exists and is ok
                                        # check for temp.json in working directory (required by slave nodes)
            tmp_file = os.path.join(current_path, "temp.json")
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
                mess_tmp_file = "Auxiliary file '"+tmp_file+"' already exists, but is cleaned up to be updated" + \
                                " for safety reasons."
                logging.info(mess_tmp_file)

        # ML 2020/06/08: Dirty workaround as long as data-splitting is done with a seperate Python-script
        #                called from the same parent Shell-/Batch-script
        #                -> work with temproary json-file in working directory
        # create or update temp.json, respectively
        md.write_destdir_jsontmp(os.path.join(md.expdir, md.expname), tmp_dir=current_path)
        
        # expand destination directory by pickle-subfolder and...
        destination_dir = os.path.join(md.expdir, md.expname, "pickle", years)
        
        # ...create directory if necessary
        if not os.path.exists(destination_dir):  # check if the Destination dir. is existing
            logging.critical('The Destination does not exist')
            logging.info('Create new destination dir')
            os.makedirs(destination_dir, exist_ok=True)
        
        with open(os.path.join(md.expdir, md.expname, "options.json"), "w") as f:
            f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    if my_rank == 0:  # node is master:
        # ==================================== Master : Directory scanner ================================= #

        print(" # ==============  Directory scanner : start    ==================# ")

        ret_dir_scanner = directory_scanner(source_dir_full)
        print(ret_dir_scanner)
        dir_detail_list = ret_dir_scanner[0]
        sub_dir_list = ret_dir_scanner[1]
        total_size_source = ret_dir_scanner[2]
        total_num_files = ret_dir_scanner[3]
        total_num_dir = ret_dir_scanner[4]

        # ===================================  Master : Load Distribution   ========================== #

        print(" # ==============  Load Distrbution  : start  ==================# ")
        
        ret_load_balancer = load_distributor(dir_detail_list, sub_dir_list, total_size_source, total_num_files,
                                             total_num_dir, p)
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
            # print('Warning:', message_in)
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

        # ========================================== Slave : Send / Receive ========================================= #
        message_in = comm.recv()

        if message_in is None:  # in case more than number of the dir. processor is assigned todo Tag it!
            message_out = ('Node', str(my_rank), 'is idle')
            comm.send(message_out, dest=0)

        else:  # if the Slave node has joblist to do
            job_list = message_in.split(';')

            for job_count in range(0, len(job_list)):
                job = job_list[job_count]  # job is the name of the directory(ies) assigned to slave_node
                # grib_2_netcdf(rot_grid,source_dir, destination_dir, job)
                if rsync_status == 1:
                    # ML 2020/06/09: workaround to get correct destination_dir obtained by the master node
                    destination_dir = MetaData.get_destdir_jsontmp(tmp_dir=current_path)
                    process_data = PreprocessNcToPkl(source_dir, destination_dir, years, job, tar_dom, vars1)
                    process_data()

                # Send : the finish of the sync message back to master node
                message_out = ('Node:', str(my_rank), 'finished :', "", '\r\n')
                comm.send(message_out, dest=0)

    MPI.Finalize()


if __name__ == "__main__":
    main()
