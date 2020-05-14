#!/bin/bash -x


mpirun -np 4 python ../../workflow_parallel_frame_prediction/DataExtraction/mpi_stager_v2.py --source_dir /home/b.gong/data_era5/2017/ --destination_dir /home/${USER}/extractedData/2017
