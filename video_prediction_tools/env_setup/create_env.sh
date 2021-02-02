#!/usr/bin/env bash
#
# __authors__ = Bing Gong, Michael Langguth
# __date__  = '2020_09_10'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for ambs-project
# or to simply activate it.
# It also converts the (Batch) runscript templates to executable runscripts.
# Note, that you may pass an experiment identifier as second argument (default 'exp1') to this runscript
# which will also be used as suffix in the executable runscripts.
# **************** Description ****************
#
# some first sanity checks
if [[ ${BASH_SOURCE[0]} == ${0} ]]; then
  echo "ERROR: 'create_env.sh' must be sourced, i.e. execute by prompting 'source create_env.sh [virt_env_name]'"
  exit 1
fi

# from now on, just return if something unexpected occurs instead of exiting
# as the latter would close the terminal including logging out
if [[ ! -n "$1" ]]; then
  echo "ERROR: Provide a name to set up the virtual environment, i.e. execute by prompting 'source create_env.sh [virt_env_name]"
  return
fi

HOST_NAME=`hostname`
ENV_NAME=$1
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
ENV_DIR=${WORKING_DIR}/${ENV_NAME}

# further sanity checks:
# * ensure execution from env_setup-directory
# * check if virtual env has already been set up

if [[ "${EXE_DIR}" != "env_setup"  ]]; then
  echo "ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  return
fi

if [[ -d ${ENV_DIR} ]]; then
  echo "Virtual environment has already been set up under ${ENV_DIR}. The present virtual environment is activated now."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

# add personal email-address to Batch-scripts
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
    # load modules and check for their availability
    echo "***** Checking modules required during the workflow... *****"
    source ${ENV_SETUP_DIR}/modules_preprocess.sh
    source ${ENV_SETUP_DIR}/modules_train.sh

elif [[ "${HOST_NAME}" == "zam347" ]]; then
    unset PYTHONPATH
fi

if [[ "$ENV_EXIST" == 0 ]]; then
  # Activate virtual environmen and install additional Python packages.
  echo "Configuring and activating virtual environment on ${HOST_NAME}"
    
  python3 -m venv $ENV_DIR
  
  activate_virt_env=${ENV_DIR}/bin/activate
  echo ${activate_virt_env}
  
  source ${activate_virt_env}
  
  # install some requirements and/or check for modules
  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
    # check module availability for the first time on known HPC-systems
    echo "***** Start installing additional Python modules with pip... *****"
    pip3 install --no-cache-dir --ignore-installed -r ${ENV_SETUP_DIR}/requirements.txt
    #pip3 install --user netCDF4
    #pip3 install --user numpy
  elif [[ "${HOST_NAME}" == "zam347" ]]; then
    echo "***** Start installing additional Python modules with pip... *****"
    pip3 install --upgrade pip
    pip3 install -r ${ENV_SETUP_DIR}/requirements.txt
    pip3 install  mpi4py 
    pip3 install netCDF4
    pip3 install  numpy
    pip3 install h5py
    pip3 install tensorflow-gpu==1.13.1
  fi

  # expand PYTHONPATH...
  export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH >> ${activate_virt_env}
  #export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:$PYTHONPATH
  export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH >> ${activate_virt_env}

  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
     export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH >> ${activate_virt_env}
  fi
  # ...and ensure that this also done when the 
  echo "" >> ${activate_virt_env}
  echo "# Expand PYTHONPATH..." >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/utils/:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH " >> ${activate_virt_env}
  #export PYTHONPATH=/p/home/jusers/${USER}/juwels/.local/bin:\$PYTHONPATH
  echo "export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:\$PYTHONPATH" >> ${activate_virt_env}

  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
    echo "export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:\$PYTHONPATH" >> ${activate_virt_env}
  fi
  info_str= "Virtual environment ${ENV_DIR} has been set up successfully."
elif [[ "$ENV_EXIST" == 1 ]]; then
  # activating virtual env is suifficient
  source ${ENV_DIR}/bin/activate
  info_str= "Virtual environment ${ENV_DIR} has been activated successfully."
fi

echo "******************************************** NOTE ********************************************"
echo ${info_str}
echo "Make use of config_runscript.py to generate customized runscripts of the workflow steps."
echo "******************************************** NOTE ********************************************"
