#!/bin/bash -x

#User's input : your virtual enviornment name
VIRT_ENV_NAME=venv_test

echo "Activating virtual environment..."
source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate

#select years and variables for dataset and define target domain 
years=( "2007" )
variables=( "2t"  )
sw_corner=( 10 20)
nyx=( 40 40 )

#your source dir and target dir
source_dir=/home/b.gong/data_era5
destination_dir=/home/b.gong/preprocessed_data

#The number of nodes should be equal to the number of 1 preprcessed folder plus 1
n_nodes=3

for year in "${years[@]}"; do
  echo "start preprocessing data for year ${year}"
	mpirun -n ${n_nodes} python ../main_scripts/main_preprocess_data_step1.py \
        --source_dir ${source_dir} --destination_dir ${destination_dir} --years "${year}" \
       	--vars  "${variables[0]}" \
       	--sw_corner "${sw_corner[0]}" "${sw_corner[1]}" --nyx "${nyx[0]}" "${nyx[1]}"
done




