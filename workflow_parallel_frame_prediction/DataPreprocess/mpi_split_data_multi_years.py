from mpi4py import MPI
import argparse
from process_netCDF_v2 import *
from metadata import MetaData
import json

#add parser arguments
parser = argparse.ArgumentParser()
#parser.add_argument("--source_dir", type=str, default="/p/scratch/deepacf/bing/extractedData/")
parser.add_argument("--destination_dir","-dest",dest="destination_dir",type=str, default="/p/scratch/deepacf/bing/processData_size_64_64_3_3t_norm")
parser.add_argument("--varnames","-vars",dest="varnames", nargs = '+')
#parser.add_argument("--partition","-part",dest="partition",type=json.loads)
#                    help="--partition allows to control the splitting of the processed data in training, test and validation data. Pass a dictionary-like string.")

args = parser.parse_args()
# ML 2020/06/08: Dirty workaround as long as data-splitting is done with this seperate Python-script 
#                called from the same parent Shell-/Batch-script as 'mpi_stager_v2_process_netCDF.py'
target_dir = os.path.join(MetaData.get_destdir_jsontmp(),"hickle")
varnames = args.varnames

#partition = args.partition
#all_keys  = partition.keys()
#for key in all_keys:
#    print(partition[key]) 

cv ={}
partition1 = {
            "train":{
                #"2222":[1,2,3,5,6,7,8,9,10,11,12],
                #"2010_1":[1,2,3,4,5,6,7,8,9,10,11,12],
                #"2012":[1,2,3,4,5,6,7,8,9,10,11,12],
                #"2013_complete":[1,2,3,4,5,6,7,8,9,10,11,12],
                #"2015":[1,2,3,4,5,6,7,8,9,10,11,12],
                #"2017":[1,2,3,4,5,6,7,8,9,10,11,12]
                "2015":[1,2,3,4,5,6,7,8,9,10,11,12]
                 },
            "val":
                {"2016":[1,2,3,4,5,6,7,8,9,10,11,12]
                 },
            "test":
                {"2017":[1,2,3,4,5,6,7,8,9,10,11,12]
                 }
            }





#cv["1"] = partition1
#cv2["2"] = partition2
# ini. MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()  # rank of the node
p = comm.Get_size()  # number of assigned nods
if my_rank == 0:  # node is master
    split_data_multiple_years(target_dir=target_dir,partition=partition1,varnames=varnames)
else:
    pass
