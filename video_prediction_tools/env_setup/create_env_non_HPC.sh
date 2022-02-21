#!/usr/bin/env bash
#
# __authors__ = Bing Gong
# __date__  = '2022_02_20'

unset PYTHONPATH

ENV_NAME=$1
THIS_DIR="$(pwd)"
WORKING_DIR="$(dirname "$THIS_DIR")"
VENV_BASE=${WORKING_DIR}/virtual_envs
VENV_DIR=${WORKING_DIR}/virtual_envs/${ENV_NAME}


# check if directory to virtual environment is parsed
if [ -z "$1" ]; then
  echo "ERROR: Provide a name to set up the virtual environment."
  return
fi


#Create virtual enviornment
if ! [[ -d "${VENV_BASE}" ]]; then
	mkdir "${VENV_BASE}"
        echo "Installing virtualenv under ${VENV_BASE}..."
        cd "${VENV_BASE}"
        python3 -m virtualenv -p python3 ${ENV_NAME} 
	#activate source directory
        source ${VENV_DIR}/bin/activate
fi

#Install site packages
pip install --no-cache-dir -r requirements_non_HPC.txt
echo "The site-packages is installed for non_HPC users"

## Add modules from the project
unset PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/postprocess:$PYTHONPATH

# get back to basic directory
cd "${WORKING_DIR}" || exit


