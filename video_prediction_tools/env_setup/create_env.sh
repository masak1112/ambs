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
        if [[ $argin == *"-tf_container="* ]]; then
          TF_CONTAINER_NAME=${argin#"-tf_container="}
        fi
        if [[ $argin == *"-l_nocontainer"* ]]; then
          bool_container=0
        fi
        if [[ $argin == *"-l_nohpc"* ]]; then
          bool_hpc=0
        fi
    done
    if [[ -z "${bool_container}" ]]; then
        bool_container=1
    fi
    if [[ -z "${bool_hpc}" ]]; then
        bool_hpc=1
    fi
    # in case that no TF-container is set manually, set the default
    if [[ -z "${TF_CONTAINER_NAME}" ]]; then
      TF_CONTAINER_NAME="tensorflow_21.09-tf1-py3.sif"
    fi
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
  check_argin ${@:2}                 # sets further variables
fi

# set some variables
HOST_NAME="$(hostname)"
ENV_NAME=$1
THIS_DIR="$(pwd)"
WORKING_DIR="$(dirname "$THIS_DIR")"
EXE_DIR="$(basename "$THIS_DIR")"
ENV_DIR=${WORKING_DIR}/virtual_envs/${ENV_NAME}
TF_CONTAINER=${WORKING_DIR}/HPC_scripts/${TF_CONTAINER_NAME}
if [[ ${bool_hpc} == 0 ]]; then
  TF_CONTAINER=${WORKING_DIR}/no_HPC_scripts/${TF_CONTAINER_NAME}
fi 

## perform sanity checks

modules_purge=""
if [[ ! -f ${TF_CONTAINER} ]] && [[ ${bool_container} == 1 ]]; then
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
  if [[ ${bool_container} == 0 ]]; then
    echo "Execution without container. Please ensure that you fulfill the software requirements for Preprocessing."
    if [[ ${bool_hpc} == 1 ]]; then
      echo "Make use of modules provided on your HPC-system if possible, i.e. adapt modules_preprocess.sh and modules_train.sh."
    fi
  fi
  if [[ ${bool_hpc} == 0 ]]; then
    echo "Running on a non-HPC system. Ensure that you fulfill the software requirements on your machine, e.g. CDO."
  fi
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

  if [[ ${bool_container} == 1 ]]; then
    if [[ ${bool_hpc} == 1 ]]; then
      module purge
    fi
    singularity exec --nv "${TF_CONTAINER}" ./install_venv_container.sh "${ENV_DIR}"
  
    info_str="Virtual environment ${ENV_DIR} has been set up successfully."
  else
    if [[ ${bool_hpc} == 1 ]]; then
      source ${THIS_DIR}/modules_train.sh
    fi
    unset PYTHONPATH
    ./install_venv.sh "${ENV_DIR}"

    # Activate virtual environment again
    source "${ENV_DIR}/bin/activate"

    if [[ ${bool_hpc} == 0 ]]; then
      pip3 install --no-cache-dir tensorflow==1.13.1
    fi
  fi
elif [[ "$ENV_EXIST" == 1 ]]; then
  info_str="Virtual environment ${ENV_DIR} already exists."
fi

## load modules (for running runscript-generator...
echo "${info_str}"
if [[ ${bool_hpc} == 1 ]]; then
  echo "Load modules to enable running of runscript generator '${ENV_DIR}'."
  source ${THIS_DIR}/modules_preprocess+extract.sh
fi

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
