#!/usr/bin/env bash

if [[ ! -n "$1" ]]; then
  echo "Provide the user name, which will be taken as folder name"
  exit 1
fi

FOLDER_NAME=$1
WORKING_DIR=/p/project/deepacf/deeprain/${FOLDER_NAME}/Video_Prediction_SAVP
ENV_DIR=${WORKING_DIR}/env_setup

source ${ENV_DIR}/modules.sh
# Install additional Python packages.
pip3 install --ignore-installed -r requirements.txt
#pip3 install --user netCDF4
#pip3 install --user numpy

#Copy the hickle package from bing's account
cd ${ENV_DIR}
cp  -r /p/project/deepacf/deeprain/bing/hickle .
export PYTHONPATH=${ENV_DIR}/hickle/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/lpips-tensorflow:$PYTHONPATH


