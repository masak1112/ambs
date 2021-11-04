#!/usr/bin/env bash

# basic directory variables
ENV_SETUP_DIR=`pwd`
WORKING_DIR="$(dirname "$ENV_SETUP_DIR")"
EXE_DIR="$(basename "$ENV_SETUP_DIR")"
VENV_DIR=$WORKING_DIR/$1/$1
shift                     # replaces $1 by $2, so that $@ does not include the name of the virtual environment anymore

# sanity checks
if [[ "${EXE_DIR}" != "HPC_scripts"  ]]; then
  echo "ERROR: Run the setup-script for the enviornment from the HPC_scripts-directory!"
  exit
fi

if ! [[ -d "${VENV_DIR}" ]]; then
   echo "ERROR: Could not found virtual environment under ${VENV_DIR}!"
   exit
fi

#expand PYHTONPATH
# Include site-packages from virtual environment...
unset PYTHONPATH
export PYTHONPATH=${VENV_DIR}/lib/python3.8/site-packages/:$PYTHONPATH
# ... dist-packages from container singularity...
#export PYTHONPATH=/usr/local/lib/python3.8/dist-packages:$PYTHONPATH
# ... and modules from this project
export PYTHONPATH=${WORKING_DIR}:$PYTHONPATH
export PYTHONPATH=${WORKING_DIR}/utils:$PYTHONPATH

# Control
echo "****** Check PYTHONPATH *****"
echo $PYTHONPATH
# MPI related environmental variables
export PMIX_SECURITY_MODE="native"     # default would include munge which is unavailable

$@
