#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# select years for dataset
declare -a years=(
                 "2017"
                  )

max_year=`echo "${years[*]}" | sort -nr | head -n1`
min_year=`echo "${years[*]}" | sort -nr | tail -n1`
# set some paths
# note, that destination_dir is used during runtime to set a proper experiment directory
exp_id=xxx                                          # experiment identifier is set by 'generate_workflow_runscripts.sh'
source_dir=${SAVE_DIR}/extractedData
destination_dir=${SAVE_DIR}/preprocessedData/era5-Y${min_year}to${max_year}M01to12
script_dir=`pwd`

for year in "${years[@]}";
    do
        echo "Year $year"
        echo "source_dir ${source_dir}/${year}"
        mpirun -np 2 python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
         --source_dir ${source_dir} -scr_dir ${script_dir} -exp_dir ${exp_id} \
         --destination_dir ${destination_dir} --years ${years} --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710
    done
python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500




