#!/usr/bin/env bash
#
# __authors__ = Bing Gong, Michael Langguth
# __date__  = '2020_01_15'
# __last_update__  = '2021_10_28' by Michael Langguth
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for AMBS-project
# The name of the virtual environment is controlled by the first parsed argument.
# It also setups the (Batch) runscript templates to customized runscripts (to be used by generate_runscript.py)
# Note that the basic output directory for the workflow may be set may parsing -base_dir [my_dir].
# **************** Description ****************
#
# **************** Auxiliary functions ****************
check_argin() {
# Handle input arguments and check if one is equal to -lcontainer
# Can also be used to check for non-positional arguments (such as -exp_id=*, see commented lines)
    for argin in "$@"; do
        if [[ $argin == *"-base_dir="* ]]; then
          base_outdir=${argin#"-base_dir="}
        fi
    done
}

# **************** Auxiliary functions ****************

# **************** Actual script ****************
# some first sanity checks
if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
  echo "ERROR: 'create_env.sh' must be sourced, i.e. execute by prompting 'source create_env.sh [virt_env_name]'"
  exit 1
fi

# from now on, just return if something unexpected occurs instead of exiting
# as the latter would close the terminal including logging out
if [[ -z "$1" ]]; then
  echo "ERROR: Provide a name to set up the virtual environment, i.e. execute by prompting 'source create_env.sh [virt_env_name]"
  return
fi

if [[ "$#" -gt 1 ]]; then
  check_argin ${@:2}                 # sets base_outdir if provided
fi

# set some variables
HOST_NAME="$(hostname)"
ENV_NAME=$1
THIS_DIR="$(pwd)"
WORKING_DIR="$(dirname "$THIS_DIR")"
EXE_DIR="$(basename "$THIS_DIR")"
ENV_DIR=${WORKING_DIR}/virtual_envs/${ENV_NAME}
TF_CONTAINER=${WORKING_DIR}/HPC_scripts/tensorflow_21.09-tf1-py3.sif

## perform sanity checks

modules_purge=""
if [[ ! -f ${TF_CONTAINER} ]]; then
  echo "ERROR: Cannot find required TF1.15 container image '${TF_CONTAINER}'."
  return
fi

# further sanity checks:
# * ensure execution from env_setup-directory
# * check host
# * check if virtual env has already been set up

if [[ "${EXE_DIR}" != "env_setup"  ]]; then
  echo "ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  return
fi

if ! [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin*  ]]; then
  echo "ERROR: AMBS-workflow is currently only supported on the Juelich HPC-systems HDF-ML, Juwels and Juwels Booster"
  return
  # unset PYTHONPATH on every other machine that is not a known HPC-system
  # unset PYTHONPATH
fi

if [[ -d ${ENV_DIR} ]]; then
  echo "Virtual environment has already been set up under ${ENV_DIR}. The present virtual environment will be activated now."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## set up virtual environment if required
if [[ "$ENV_EXIST" == 0 ]]; then
  # Activate virtual environment and install additional Python packages.
  echo "Configuring and activating virtual environment on ${HOST_NAME}"
  
  module purge 
  singularity exec --nv "${TF_CONTAINER}" ./install_venv_container.sh "${ENV_DIR}"
  
  info_str="Virtual environment ${ENV_DIR} has been set up successfully."
elif [[ "$ENV_EXIST" == 1 ]]; then
  info_str="Virtual environment ${ENV_DIR} already exists."
fi

## load modules (for running runscript-generator...
echo "${info_str}"
echo "Load modules to enable running of runscript generator '${ENV_DIR}'."
source ${THIS_DIR}/modules_preprocess+extract.sh

## ... and prepare runscripts
echo "Set up runscript template for user ${USER}..."
if [[ -z "${base_outdir}" ]]; then
  "${WORKING_DIR}"/utils/runscript_generator/setup_runscript_templates.sh
else
  "${WORKING_DIR}"/utils/runscript_generator/setup_runscript_templates.sh "${base_outdir}"
fi

echo "******************************************** NOTE ********************************************"
echo "Make use of generate_runscript.py to generate customized runscripts of the workflow steps."
echo "******************************************** NOTE ********************************************"
