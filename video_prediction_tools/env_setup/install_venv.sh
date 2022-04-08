#!/usr/bin/env bash
#
# __authors__ = Bing Gong, Michael Langguth
# __date__  = '2022_02_28'
# __last_update__  = '2022_02_28' by Michael Langguth
#
# **************** Description ****************
# This auxiliary script sets up the virtual environment OUTSIDE singularity container.
# **************** Description ****************

# set some basic variables
BASE_DIR="$(pwd)"
VENV_DIR=$1
VENV_NAME="$(basename "${VENV_DIR}")"
VENV_BASE="$(dirname "${VENV_DIR}")"
WORKING_DIR="$(dirname "${VENV_BASE}")"
VENV_REQ=${BASE_DIR}/requirements_nocontainer.txt

# sanity checks

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

# get Python-version
PYTHON_VERSION=$(python3 -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')
unset PYTHONPATH

# create or change to  base directory for virtual environment (i.e. where the virtualenv-module is placed)
if ! [[ -d "${VENV_BASE}" ]]; then
  mkdir "${VENV_BASE}"
  # Install virtualenv in this directory
  echo "Installing virtualenv under ${VENV_BASE}..."
  pip3 install --target="${VENV_BASE}/" virtualenv
  # Change into the base-directory of virtual environments...
  cd "${VENV_BASE}" || return
else
  # Change into the base-directory of virtual environments...
  cd "${VENV_BASE}" || return
  if ! python3 -m virtualenv --version >/dev/null; then
    echo "WARNING: Base directory for virtual environment exists, but virtualenv-module is unavailable."
    echo "Try installing virtualenv."
    pip3 install --target="${VENV_BASE}/" virtualenv
  fi
  echo "Virtualenv is already installed."
fi


# Set-up virtual environment in base directory for virtual environments
python3 -m virtualenv "${VENV_NAME}"
# Activate virtual environment and install required packages
echo "Activating virtual environment ${VENV_NAME} to install required Python modules..."
ACT_VENV="${VENV_DIR}/bin/activate"
source "${VENV_DIR}/bin/activate"
# set PYTHONPATH...
export PYTHONPATH=""
export PYTHONPATH=${WORKING_DIR}/virtual_envs/${VENV_NAME}/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/model_modules:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/postprocess:$PYTHONPATH
# ... also ensure that PYTHONPATH is appended when activating the virtual environment...
echo 'export PYTHONPATH='"" >> ${ACT_VENV}
echo 'export PYTHONPATH='${WORKING_DIR}'/virtual_envs/'${VENV_NAME}'/lib/python'${PYTHON_VERSION}'/site-packages:$PYTHONPATH' >> ${ACT_VENV}
echo 'export PYTHONPATH='${WORKING_DIR}':$PYTHONPATH' >> ${ACT_VENV}
echo 'export PYTHONPATH='${WORKING_DIR}'/utils:$PYTHONPATH' >> ${ACT_VENV}
echo 'export PYTHONPATH='${WORKING_DIR}'/model_modules:$PYTHONPATH' >> ${ACT_VENV}
echo 'export PYTHONPATH='${WORKING_DIR}'/postprocess:$PYTHONPATH' >> ${ACT_VENV}
# ... install requirements
pip3 install --no-cache-dir -r "${VENV_REQ}"

# get back to basic directory
cd "${BASE_DIR}" || exit



