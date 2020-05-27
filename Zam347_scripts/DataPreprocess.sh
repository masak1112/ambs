#!/bin/bash -x


source_dir=/home/$USER/extractedData
destination_dir=/home/$USER/preprocessedData/era5-Y2015toY2017M01to12-64x64-74d00N71d00E-T_MSL_gph500/hickle
declare -a years=("2017")

for year in "${years[@]}";
    do
        echo "Year $year"
        echo "source_dir ${source_dir}/${year}"
        mpirun -np 2 python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
         --source_dir ${source_dir}/${year}/ \
         --destination_dir ${destination_dir}/${year}/ --vars T2 MSL gph500 --lat_s 138 --lat_e 202 --lon_s 646 --lon_e 710
    done
python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir}




