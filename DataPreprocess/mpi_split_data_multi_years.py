from mpi4py import MPI
import argparse
from process_netCDF_v2 import *

parser = argparse.ArgumentParser()
#parser.add_argument("--source_dir", type=str, default="/p/scratch/deepacf/bing/extractedData/")
parser.add_argument("--destination_dir", type=str, default="/p/scratch/deepacf/bing/processData_size_64_64_3_3t_norm")
args = parser.parse_args()
target_dir = args.destination_dir


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
    split_data_multiple_years(target_dir=target_dir,partition=partition)
else:
    pass
