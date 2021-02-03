#!/usr/bin/env bash
#
# __authors__ = Bing Gong, Michael Langguth
# __date__  = '2020_01_15'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for AMBS-project
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

if [[ -n "$2" ]]; then
  exp_id=$2
else
  exp_id=""
fi

# list of (Batch) scripts used for the steps in the workflow
# !!! Expects that a template named [script_name]_template.sh exists                   !!!
# !!! Only create runscripts for data extraction and preprocessing step 1.             !!!
# !!! For the rest, make use of config_runscript.sh                                    !!!
workflow_scripts=(data_extraction_era5 preprocess_data_era5_step1)

HOST_NAME=`hostname`
ENV_NAME=$1
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
ENV_DIR=${WORKING_DIR}/${ENV_NAME}

# list of (Batch) scripts used for the steps in the workflow
# !!! Expects that a template named [script_name]_template.sh exists!!!
if [[ "${HOST_NAME}" == jwlogin2[1-4]* ]]; then
  echo "******************************************** NOTE ********************************************"
  echo "                Make use of dedicated Horovod-related working branches only!!!                "
  echo "******************************************** NOTE ********************************************"
  workflow_scripts=()
  # another sanity check for Juwels Booster -> ensure running singularity
  if [[ -z "${SINGULARITY_NAME}" ]]; then
    echo "ERROR: create_env.sh must be executed in a running singularity on Juwels Booster."
    echo "Thus, execute 'singularity shell [my_docker_image]' first!"
    return
  fi
else
  workflow_scripts=(data_extraction_era5 preprocess_data_era5_step1 preprocess_data_era5_step2 train_model_era5 visualize_postprocess_era5 preprocess_data_moving_mnist train_model_moving_mnist visualize_postprocess_moving_mnist)
fi

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
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *juwels* ]]; then
  if [[ "${HOST_NAME}" == jwlogin2[1-4]* ]]; then  
    # on Juwels Booster, we are in a container environment -> loading modules is not possible	  
    echo "***** Note for Juwels Booster! *****"
    echo "Already checked the required modules?"
    echo "To do so, run 'source modules_train.sh' after exiting the singularity."
    echo "***** Note for Juwels Booster! *****"
  else
    # load modules and check for their availability
    echo "***** Checking modules required during the workflow... *****"
    source ${ENV_SETUP_DIR}/modules_preprocess.sh purge
    source ${ENV_SETUP_DIR}/modules_train.sh
  fi
else 
  # unset PYTHONPATH on every other machine that is not a known HPC-system	
  unset PYTHONPATH
fi

if [[ "$ENV_EXIST" == 0 ]]; then
  # Activate virtual environmen and install additional Python packages.
  echo "Configuring and activating virtual environment on ${HOST_NAME}"
    
  python3 -m venv $ENV_DIR
  
  activate_virt_env=${ENV_DIR}/bin/activate

  echo "Entering virtual environment ${ENV_DIR} to install required Python modules..."
  source ${activate_virt_env}
  
  # install some requirements and/or check for modules
  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *juwels* ]]; then
    # Install packages depending on host
    echo "***** Start installing additional Python modules with pip... *****"
    req_file=${ENV_SETUP_DIR}/requirements.txt 
    if [[ "${HOST_NAME}" == jwlogin2[1-4]* ]]; then req_file=${ENV_SETUP_DIR}/requirements_booster.txt; fi
    
    pip3 install --no-cache-dir -r ${req_file}
  else
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
  export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:$PYTHONPATH >> ${activate_virt_env}
  export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH >> ${activate_virt_env}

  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *juwels* ]]; then
     export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:$PYTHONPATH >> ${activate_virt_env}
     if [[ "${HOST_NAME}" == jwlogin2[1-4]* ]]; then
       export PYTONPATH=/usr/locali/lib/python3.6/dist-packages:$PYTHONPATH
     fi
  fi
  # ...and ensure that this also done when the 
  echo "" >> ${activate_virt_env}
  echo "# Expand PYTHONPATH..." >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/utils/:\$PYTHONPATH" >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH " >> ${activate_virt_env}
  echo "export PYTHONPATH=${WORKING_DIR}/external_package/lpips-tensorflow:\$PYTHONPATH" >> ${activate_virt_env}

  if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *juwels* ]]; then
    echo "export PYTHONPATH=${ENV_DIR}/lib/python3.6/site-packages:\$PYTHONPATH" >> ${activate_virt_env}
     if [[ "${HOST_NAME}" == jwlogin2[1-4]* ]]; then
       echo "export PYTONPATH=/usr/locali/lib/python3.6/dist-packages:\$PYTHONPATH" >> ${activate_virt_env}
     fi
  fi
elif [[ "$ENV_EXIST" == 1 ]]; then
  echo "ERROR: Virtual environment ${ENV_NAME} already exists, please choose another name or delete the existing one."
  return
fi
# Finish by creating runscripts
 # After checking and setting up the virt env, create user-specific runscripts for all steps of the workflow
if [[ "${HOST_NAME}" == *hdfml* || "${HOST_NAME}" == *juwels* ]]; then
  script_dir=../HPC_scripts
else
  script_dir=../Zam347_scripts
fi

echo "***** Creating Batch-scripts for data extraction and prepropcessing step 1... *****"
for wf_script in "${workflow_scripts[@]}"; do
  curr_script=${script_dir}/${wf_script}
  if [[ -z "${exp_id}" ]]; then
    ./generate_workflow_runscripts.sh ${curr_script} ${ENV_NAME}
  else
    ./generate_workflow_runscripts.sh ${curr_script}  ${ENV_NAME} -exp_id=${exp_id}
  fi
done
echo "******************************************** NOTE ********************************************"
echo "Runscripts for the remaining workflow steps can be generated with config_runscript.py!        "

# finally deactivate virtual environment and clean up loaded modules (if we are not on Juwels)
deactivate
if [[ "${HOST_NAME}" == *hdfml* || "${HOST_NAME}" == *juwels* ]] && [[ "${HOST_NAME}" != jwlogin2[1-4]* ]]; then
  module --force purge
fi

