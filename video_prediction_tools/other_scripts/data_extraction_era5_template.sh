#!/bin/bash -x

#User's input : your virtual enviornment name
VIRT_ENV_NAME=venv_test

echo "Activating virtual environment..."
source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate

# Declare path-variables (dest_dir will be set and configured automatically via generate_runscript.py)
source_dir=/my/path/to/era5
destination_dir=/my/path/to/extracted/data
varmap_file=/my/path/to/varmapping/file

years=( "2007" )

#The number of nodes should be equal to the number of 1 preprcessed folder plus 1
n_nodes=3

# Run data extraction
for year in "${years[@]}"; do
  echo "Perform ERA5-data extraction for year ${year}"
  python ../main_scripts/main_data_extraction.py  --source_dir ${source_dir} --target_dir ${destination_dir} \
                                                       --year ${year} --varslist_path ${varmap_file}
done




