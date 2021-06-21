"""
Driver for preprocessing step 2 which parses the input arguments from the runscript
and performs parallelization with OpenMPI.
"""
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"

# import modules
import os
import argparse
from mpi4py import MPI
from general_utils import get_unique_vars
from statistics import Calc_data_stat
from data_preprocess.preprocess_data_step2 import *
import warnings


def main():

    method="main_preprocess_data_step2"

    parser = argparse.ArgumentParser()
    parser.add_argument("-source_dir", type=str)
    parser.add_argument("-dest_dir", type=str)
    parser.add_argument("-sequence_length", type=int, default=20)
    parser.add_argument("-sequences_per_file", type=int, default=20)
    args = parser.parse_args()
    input_dir = args.source_dir
    ins = ERA5Pkl2Tfrecords(input_dir=input_dir,
                            dest_dir=args.dest_dir,
                            sequence_length = args.sequence_length,
                            sequences_per_file=args.sequences_per_file)
    
    years, months,years_months = ins.get_years_months()
    # ini. MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()  # rank of the node
    p = comm.Get_size()  # number of assigned nodes
    if p < 2:
        raise ValueError("%{0}: Preprocessing step 2 must be assigned to at least two tasks.".format(method))
  
    if my_rank == 0:
        # retrieve final statistics first (not parallelized!)
        # some preparatory steps
        stat_dir = os.path.dirname(input_dir)
        varnames = ins.vars_in
    
        vars_uni, varsind, nvars = get_unique_vars(varnames)
        stat_obj = Calc_data_stat(nvars)                            # init statistic-instance
    
        # loop over whole data set (training, dev and test set) to collect the intermediate statistics
        print("%{0}: Start collecting statistics from the whole dataset to be processed...".format(method))
       
        for year in years:
            file_dir = os.path.join(input_dir, year)
            for month in months:
                if os.path.isfile(os.path.join(file_dir, "stat_" + '{0:02}'.format(month) + ".json")):
                    # process stat-file:
                    stat_obj.acc_stat_master(file_dir, int(month))  # process monthly statistic-file
                else:
                    warnings.warn("%{0}: The statistic file for year {1}, month {2} does not exist".format(method, year, month))
        # finalize statistics and write to json-file
        stat_obj.finalize_stat_master(vars_uni)
        stat_obj.write_stat_json(stat_dir)

        # organize parallelized partioning 
        real_years_months = []
        for i in range(len(years)):
            year = years[i]
            for month in years_months[i]:
                year_month = "Y_{}_M_{}".format(year, month)
                real_years_months.append(year_month)
 
        broadcast_lists = [list(years), real_years_months]

        for nodes in range(1, p):
            comm.send(broadcast_lists, dest=nodes)
           
        message_counter = 1
        while message_counter <= p-1:
            message_in = comm.recv()
            message_counter = message_counter + 1 
            print("%{0}: Message in from worker: {1} ".format(method, message_in))
 
    else:
        message_in = comm.recv()
        print("%{0}: Message from master to rank {1}: {2} ".format(method, my_rank, message_in))
        
        years = list(message_in[0])
        real_years_months = message_in[1] 
   
        for year in years:
            year_rank = "Y_{}_M_{}".format(year, my_rank)
            if year_rank in real_years_months:
                # Initilial instance
                ins2 = ERA5Pkl2Tfrecords(input_dir=input_dir,
                                         dest_dir=args.dest_dir,
                                         sequence_length = args.sequence_length,
                                         sequences_per_file=args.sequences_per_file)
                # create the tfrecords-files
                ins2.read_pkl_and_save_tfrecords(year=year, month=my_rank)
                print("%{0}: Year {1} finished".format(method, year))
            else:
                print("%{0}: {1} is not in the datasplit_dic, will skip the process".format(method, year_rank))
        message_out = ("Node:", str(my_rank), "finished", "", "\r\n")
        print("%{0}: Message out for worker: {1}".format(method, message_out))
        comm.send(message_out, dest=0)

    MPI.Finalize()

   
if __name__ == '__main__':
    main()
