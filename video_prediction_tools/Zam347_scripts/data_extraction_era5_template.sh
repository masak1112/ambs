#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

mpirun -np 4 python ../../workflow_parallel_frame_prediction/DataExtraction/mpi_stager_v2.py --source_dir /home/b.gong/data_era5/2017/ --destination_dir /home/${USER}/extractedData/2017
