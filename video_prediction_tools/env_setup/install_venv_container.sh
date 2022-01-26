#!/usr/bin/env bash
#
# __authors__ = Bing Gong, Michael Langguth
# __date__  = '2021_10_28'
# __last_update__  = '2022_01_26' by Michael Langguth
#
# **************** Description ****************
# This auxiliary script sets up the virtual environment within a singularity container.
# **************** Description ****************

# set some basic variables
BASE_DIR="$(pwd)"
VENV_BASE=$1
VENV_NAME="$(basename "${VENV_BASE}")"
VENV_DIR=${VENV_BASE}/${VENV_NAME}
VENV_REQ=${BASE_DIR}/requirements.txt

# sanity checks
# check if we are running in a container
if [ -z "${SINGULARITY_NAME}" ]; then
  echo "ERROR: install_venv_container.sh must be called within a running singularity container."
  return
fi

# check if directory to virtual environment is parsed
if [ -z "$1" ]; then
  echo "ERROR: Provide a name to set up the virtual environment."
  return
fi

# check if virtual environment is not already existing
if [ -d "$1" ]; then
  echo "ERROR: Target directory of virtual environment ${1} already exists. Chosse another directory path."
  return
fi

# check for requirement-file
if [ ! -f "${VENV_REQ}" ]; then
  echo "ERROR: Cannot find requirement-file '${VENV_REQ}' to set up virtual environment."
  return
fi


# create basic target directory for virtual environment
mkdir "${VENV_BASE}"
# Install virtualenv in this directory
echo "Installing virtualenv under ${VENV_BASE}..."
pip install --target="${VENV_BASE}/" virtualenv
# Change into the directory...
cd "${VENV_BASE}" || exit
# .. to set-up virtual environment therein
python -m virtualenv -p /usr/bin/python "${VENV_NAME}"
# Activate virtual environment and install required packages
echo "Actiavting virtual environment ${VENV_NAME} to install required Python modules..."
ACT_VENV="${VENV_DIR}/bin/activate"
source "${VENV_DIR}/bin/activate"
# set PYTHONPATH...
export PYTHONPATH="/usr/local/lib/python3.8/dist-packages/"
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/postprocess:$PYTHONPATH
# ... also ensure that PYTHONPATH is appended when activating the virtual environment...
echo 'export PYTHONPATH="/usr/local/lib/python3.8/dist-packages/"' >> "${ACT_VENV}"
echo 'export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH' >> ${ACT_VENV}
echo 'export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH' >> ${ACT_VENV}
echo 'export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH' >> ${ACT_VENV}
echo 'export PYTHONPATH=${WORKING_DIR}/postprocess:$PYTHONPATH' >> ${ACT_VENV}
# ... install requirements
pip install --no-cache-dir -r "${VENV_REQ}"

# get back to basic directory
cd "${BASE_DIR}" || exit



