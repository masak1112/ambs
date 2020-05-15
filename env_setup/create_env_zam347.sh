#!/usr/bin/env bash


if [[ ! -n "$1" ]]; then
  echo "Provide the env name, which will be taken as folder name"
  exit 1
fi

ENV_NAME=$1
WORKING_DIR=/home/$USER/video_prediction_savp
ENV_SETUP_DIR=${WORKING_DIR}/env_setup
ENV_DIR=${WORKING_DIR}/${ENV_NAME}
unset PYTHONPATH
#source ${ENV_SETUP_DIR}/modules.sh
# Install additional Python packages.
python3 -m venv $ENV_DIR
source ${ENV_DIR}/bin/activate
pip3 install --upgrade pip
pip3 install -r ${ENV_SETUP_DIR}/requirements.txt
#conda install mpi4py
pip3 install  mpi4py 
pip3 install netCDF4
pip3 install  numpy
pip3 install h5py
pip3 install tensorflow==1.13.1
#Copy the hickle package from bing's account
#cp  -r /p/project/deepacf/deeprain/bing/hickle ${WORKING_DIR}

#source ${ENV_SETUP_DIR}/modules.sh
#source ${ENV_DIR}/bin/activate

#export PYTHONPATH=/home/$USER/miniconda3/pkgs:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/hickle/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
#export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH
#export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/lpips-tensorflow:$PYTHONPATH


