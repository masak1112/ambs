#!/bin/bash -x


source_dir=/home/$USER/extractedData
destination_dir=/home/$USER/preprocessedData/era5-Y2017M01to02
script_dir=`pwd`

declare -a years=("2017")

for year in "${years[@]}";
    do
        echo "Year $year"
        echo "source_dir ${source_dir}/${year}"
        mpirun -np 2 python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
         --source_dir ${source_dir} -scr_dir ${script_dir} \
         --destination_dir ${destination_dir} --years ${years} --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710
    done
python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500




