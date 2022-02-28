#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment
VIRT_ENV_NAME=venv_test

if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

#select years and variables for dataset and define target domain 
years=( "2007" )
variables=( "2t"  )
sw_corner=( 10 20)
nyx=( 40 40 )

#your source dir and target dir
source_dir=/home/b.gong/data_era5
destination_dir=/home/b.gong/preprocessed_data

#The number of nodes should be equal to the number of 1 preprocessed folders plus 1
n_nodes=3

for year in "${years[@]}"; do
  echo "start preprocessing data for year ${year}"
	mpirun -n ${n_nodes} python ../main_scripts/main_preprocess_data_step1.py \
        --source_dir ${source_dir} --destination_dir ${destination_dir} --years "${year}" \
       	--vars  "${variables[0]}" \
       	--sw_corner "${sw_corner[0]}" "${sw_corner[1]}" --nyx "${nyx[0]}" "${nyx[1]}"
done




