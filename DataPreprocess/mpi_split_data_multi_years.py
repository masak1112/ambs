from mpi4py import MPI
import argparse
from process_netCDF_v2 import *
import json

#add parser arguments
parser = argparse.ArgumentParser()
#parser.add_argument("--source_dir", type=str, default="/p/scratch/deepacf/bing/extractedData/")
parser.add_argument("--destination_dir","-dest",dest="destination_dir",type=str, default="/p/scratch/deepacf/bing/processData_size_64_64_3_3t_norm")
parser.add_argument("--varnames","-vars",dest="varnames", nargs = '+')
#parser.add_argument("--partition","-part",dest="partition",type=json.loads)
#                    help="--partition allows to control the splitting of the processed data in training, test and validation data. Pass a dictionary-like string.")

args = parser.parse_args()
target_dir = os.system("echo ${dest_dir_split}")#args.destination_dir
varnames = args.varnames

#partition = args.partition
#all_keys  = partition.keys()
#for key in all_keys:
#    print(partition[key]) 

partition = {
            "train":{
                "2017":[1]
                 },
            "val":
                {"2017":[2]
                 },
            "test":
                {"2017":[2]
                 }
            }
# ini. MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()  # rank of the node
p = comm.Get_size()  # number of assigned nods
if my_rank == 0:  # node is master
    split_data_multiple_years(target_dir=target_dir,partition=partition,varnames=varnames)
else:
    pass
