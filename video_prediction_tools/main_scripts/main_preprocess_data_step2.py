"""
Driver for preprocessing step 2 which parses the input arguments from the runscript
and performs parallelization with OpenMPI.
"""
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Scarlet Stadtler,Michael Langguth"

# import modules
import os
import argparse
from mpi4py import MPI
from general_utils import get_unique_vars
from statistics import Calc_data_stat
from video_prediction.datasets.era5_dataset import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-datasplit_config", type=str, \
                        help="The path to the datasplit_config json file which contains the details of train/val/testing")
    parser.add_argument("-hparams_dict_config", type=str,\
                        help="The path to the dict that contains hparameters.", default="")
    parser.add_argument("-sequences_per_file", type=int, default=20)
    args = parser.parse_args()
    ins = ERA5Pkl2Tfrecords(input_dir=args.input_dir, output_dir=args.output_dir,
                            datasplit_config=args.datasplit_config,
                            hparams_dict_config=args.hparams_dict_config,
                            sequences_per_file=args.sequences_per_file)
    
    partition = ins.data_dict
    partition_data = partition.values()
    years, months = ins.get_years_months()
    input_dir_pkl = os.path.join(args.input_dir, "pickle")
    # ini. MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()  # rank of the node
    p = comm.Get_size()  # number of assigned nods
  
    if my_rank == 0:
        # retrieve final statistics first (not parallelized!)
        # some preparatory steps
        stat_dir_prefix = input_dir_pkl
        varnames        = ins.vars_in
    
        vars_uni, varsind, nvars = get_unique_vars(varnames)
        stat_obj = Calc_data_stat(nvars)                            # init statistic-instance
    
        # loop over whole data set (training, dev and test set) to collect the intermediate statistics
        print("Start collecting statistics from the whole dataset to be processed...")
        for split in partition.keys():
            values = partition[split]
            for year in values.keys():
                file_dir = os.path.join(stat_dir_prefix, year)
                for month in values[year]:
                    if os.path.isfile(os.path.join(file_dir, "stat_" + '{0:02}'.format(month) + ".json")):
                        # process stat-file:
                        stat_obj.acc_stat_master(file_dir, int(month))  # process monthly statistic-file
                    else:
                        raise ("The stat file does not exist:", os.path.join(file_dir, "stat_" + '{0:02}'.\
                                                                             format(month) + ".json"))
        # finalize statistics and write to json-file
        stat_obj.finalize_stat_master(vars_uni)
        stat_obj.write_stat_json(input_dir_pkl)

        # organize parallelized partioning 
        
        real_years_months = []
        for year_months in partition_data:
            print("I am here year:", year_months)
            for year in year_months.keys():
                for month in year_months[year]:
                    print("I am here month", month)
                    year_month = "Y_{}_M_{}".format(year, month)
                    real_years_months.append(year_month)
 
        broadcast_lists = [list(years), real_years_months]

        for nodes in range(1, p):
            comm.send(broadcast_lists, dest=nodes)
           
        message_counter = 1
        while message_counter <= 12:
            message_in = comm.recv()
            message_counter = message_counter + 1 
            print("Message in from slave: ", message_in)
 
    else:
        message_in = comm.recv()
        print("My rank,", my_rank)
        print("message_in", message_in)
        
        years = list(message_in[0])
        real_years_months = message_in[1] 
   
        for year in years:
            # loop over the years in the datasplit_dict which we want to process,
            # while months that are not in the datasplit_dict are skipped
            year_rank = "Y_{}_M_{}".format(year, my_rank)
            if year_rank in real_years_months:
                # Initilial instance
                ins2 = ERA5Pkl2Tfrecords(input_dir=args.input_dir, output_dir=args.output_dir,
                            datasplit_config=args.datasplit_config,
                            hparams_dict_config=args.hparams_dict_config, sequences_per_file=args.sequences_per_file)
                # create the tfrecords-files
                ins2.read_pkl_and_save_tfrecords(year=year, month=my_rank)
                print("Year {} finished", year)
            else:
                print(year_rank + " is not in the datasplit_dic, will skip the process")
        message_out = ("Node:", str(my_rank), "finished", "", "\r\n")
        print("Message out for slaves:", message_out)
        comm.send(message_out, dest=0)

    MPI.Finalize()

   
if __name__ == '__main__':
    main()
