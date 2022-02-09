#!/bin/bash -x
## Controlling Batch-job : Need input
#SBATCH --account=<Project name>
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=Data_Preprocess_step1_era5-out.%j
#SBATCH --error=Data_Preprocess_step1era5-err.%j
#SBATCH --time=04:20:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

##Load basic Python module: Need input
#module load Python


##Create and activate a virtual environment : Need input
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




