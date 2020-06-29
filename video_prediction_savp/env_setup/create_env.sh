#!/usr/bin/env bash


if [[ ! -n "$1" ]]; then
  echo "Provide the env name, which will be taken as folder name"
  exit 1
fi

ENV_NAME=$1
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
ENV_DIR=${WORKING_DIR}/${ENV_NAME}
USER_EMAIL=$(jutil user show -o json | grep email | cut -f2 -d':' | cut -f1 -d',' | cut -f2 -d'"')
echo $USER_EMAIL
#Set up global env variable "save_dir" used for define the target save path
export SAVE_DIR=/p/scratch/deepacf/video_prediction_shared_folder/


#replace the email in sbatch script with the USER_EMAIL
sed -i "s/--mail-user=.*/--mail-user=$USER_EMAIL/g" ../HPC_scripts/*.sh

source ${ENV_SETUP_DIR}/modules.sh
# Install additional Python packages.
python3 -m venv $ENV_DIR
source ${ENV_DIR}/bin/activate
pip3 install -r ${ENV_SETUP_DIR}/requirements.txt
#pip3 install --user netCDF4
#pip3 install --user numpy

source ${ENV_SETUP_DIR}/modules.sh
source ${ENV_DIR}/bin/activate

export PYTHONPATH=${WORKING_DIR}/external_package/hickle/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH
#export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:$PYTHONPATH


