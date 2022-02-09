#!/bin/bash -x
## Controlling Batch-job : Need input
#SBATCH --account=<Project name>
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=data_extraction_era5-out.%j
#SBATCH --error=data_extraction_era5-err.%j
#SBATCH --time=04:20:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

##Load basic Python module: Need input
#module load Python


##Create and activate a virtual environment: Need input 
#VENV_NAME=<my_venv>
#Python -m venv ../virtual_envs/${VENV_NAME}
#source ../virtual_envs/${VENV_NAME}/bin/activate


## Install required packages
# set PYTHONPATH...
BASE_DIR="$(pwd)"
WORKING_DIR=="$(BASE_DIR "$dir")"
export PYTHONPATH=${WORKING_DIR}/virtual_envs/${VENV_NAME}/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/postprocess:$PYTHONPATH
# ... install requirements
pip install --no-cache-dir -r ../env_setup/requirements_nonJSC.txt


# Declare path-variables (dest_dir will be set and configured automatically via generate_runscript.py)
source_dir=/my/path/to/era5
destination_dir=/my/path/to/extracted/data
varmap_file=/my/path/to/varmapping/file

years=( "2015" )

# Run data extraction
for year in "${years[@]}"; do
  echo "Perform ERA5-data extraction for year ${year}"
  srun python ../main_scripts/main_data_extraction.py  --source_dir ${source_dir} --target_dir ${destination_dir} \
                                                       --year ${year} --varslist_path ${varmap_file}
done
