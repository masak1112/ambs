#!/usr/bin/env bash

if [[ ! -n "$1" ]]; then
  echo "Provide the user name, which will be taken as folder name"
  exit 1
fi

if [[ ! -n "$2" ]]; then
  echo "Provide the env name, which will be taken as folder name"
  exit 1
fi

ENV_NAME=$2
FOLDER_NAME=$1
WORKING_DIR=/p/project/deepacf/deeprain/${FOLDER_NAME}/video_prediction_savp
ENV_SETUP_DIR=${WORKING_DIR}/env_setup
ENV_DIR=${WORKING_DIR}/${ENV_NAME}

source ${ENV_SETUP_DIR}/modules.sh
# Install additional Python packages.
python3 -m venv $ENV_DIR
source ${ENV_DIR}/bin/activate
pip3 install -r requirements.txt
#pip3 install --user netCDF4
#pip3 install --user numpy

#Copy the hickle package from bing's account
cp  -r /p/project/deepacf/deeprain/bing/hickle ${WORKING_DIR}

source ${ENV_SETUP_DIR}/modules.sh
source ${ENV_DIR}/bin/activate

export PYTHONPATH=${WORKING_DIR}/hickle/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH
#export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/lpips-tensorflow:$PYTHONPATH


