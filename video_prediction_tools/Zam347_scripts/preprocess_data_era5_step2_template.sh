#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/home/${USER}/preprocessedData/
destination_dir=/home/${USER}/preprocessedData/


python ../video_prediction/datasets/era5_dataset_v2.py ${source_dir}/hickle/splits ${destination_dir}/tfrecords -vars T2 MSL gph500 -height 128 -width 160 -seq_length 20 
