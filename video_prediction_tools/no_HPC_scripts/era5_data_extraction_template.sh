#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment
VIRT_ENV_NAME=venv_test

if [ -z "${VIRTUAL_ENV}" ]; then
   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

# select years and variables for dataset and define target domain
years=( 2017 )
months=( "all" )
var_dict='{"2t": {"sf": ""}, "tcc": {"sf": ""}, "t": {"ml": "p85000."}}'
sw_corner=(38.4 0.0)
nyx=(56 92)

# set some paths
# note, that destination_dir is adjusted during runtime based on the data
source_dir=/my/path/to/era5/data
destination_dir=/my/path/to/extracted/data

# Must be at least 2
n_nodes=3

# execute Python-script
mpirun -n ${n_nodes} python3 ../main_scripts/main_era5_data_extraction.py -src_dir "${source_dir}" \
       -dest_dir "${destination_dir}" -y "${years[@]}" -m "${months[@]}" \
       -swc "${sw_corner[@]}" -nyx "${nyx[@]}" -v "${var_dict}"






