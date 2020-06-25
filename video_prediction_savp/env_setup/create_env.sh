#!/usr/bin/env bash

if [[ ! -n "$1" ]]; then
  echo "Provide a name to set up the virtual environment"
  exit 1
fi

HOST_NAME=`hostname`
ENV_NAME=$1
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
ENV_DIR=${WORKING_DIR}/${ENV_NAME}

#sanity check -> ensure execution from env_setup-directory
if [[ "${EXE_DIR}" != "env_setup"  ]]; then
  echo "Please execute the setup-script for the virtual environment from the env_setup-directory!"
  exit 1
fi


if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
    USER_EMAIL=$(jutil user show -o json | grep email | cut -f2 -d':' | cut -f1 -d',' | cut -f2 -d'"')
    #replace the email in sbatch script with the USER_EMAIL
    sed -i "s/--mail-user=.*/--mail-user=$USER_EMAIL/g" ../HPC_scripts/*.sh

    # check module availability for the first time on known HPC-systems
    source ${ENV_SETUP_DIR}/modules.sh
elif [[ "${HOST_NAME}" == "zam347" ]]; then
    unset PYTHONPATH
fi

# Activate virtual environmen and install additional Python packages.
echo "Configuring and activating virtual environment on ${HOST_NAME}"
    
python3 -m venv $ENV_DIR
source ${ENV_DIR}/bin/activate
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
    # check module availability for the first time on known HPC-systems
    pip3 install -r ${ENV_SETUP_DIR}/requirements.txt
    #pip3 install --user netCDF4
    #pip3 install --user numpy
    
    source ${ENV_SETUP_DIR}/modules.sh
    source ${ENV_DIR}/bin/activate
elif [[ "${HOST_NAME}" == "zam347" ]]; then
    pip3 install --upgrade pip
    pip3 install -r ${ENV_SETUP_DIR}/requirements.txt
    pip3 install  mpi4py 
    pip3 install netCDF4
    pip3 install  numpy
    pip3 install h5py
    pip3 install tensorflow-gpu==1.13.1
fi

# expand environmental variable PYTHONPATH 
export PYTHONPATH=${WORKING_DIR}/external_package/hickle/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
#export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:$PYTHONPATH

if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
    export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH
fi


