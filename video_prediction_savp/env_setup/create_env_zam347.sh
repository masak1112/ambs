#!/usr/bin/env bash


if [[ ! -n "$1" ]]; then
  echo "Provide the env name, which will be taken as folder name"
  exit 1
fi

ENV_NAME=$1
WORKING_DIR=/home/$USER/ambs/video_prediction_savp
ENV_SETUP_DIR=${WORKING_DIR}/env_setup
ENV_DIR=${WORKING_DIR}/${ENV_NAME}
unset PYTHONPATH
# Install additional Python packages.
python3 -m venv $ENV_DIR
source ${ENV_DIR}/bin/activate
pip3 install --upgrade pip
pip3 install -r ${ENV_SETUP_DIR}/requirements.txt
pip3 install  mpi4py 
pip3 install netCDF4
pip3 install  numpy
pip3 install h5py
pip3 install tensorflow-gpu==1.13.1

#export PYTHONPATH=/home/$USER/miniconda3/pkgs:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/external_package/hickle/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
#export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH
#export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:$PYTHONPATH


