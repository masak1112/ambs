#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2021_02_05'
#
# **************** Description ****************
# Converts a given template workflow script (path/name has to be passed as first argument) to
# an executable workflow (Batch) script. However, use 'config_train.py' for convenience when runscripts for the
# training and postprocessing substeps should be generated.
#
# Examples:
#    ./generate_workflow_scripts.sh ../HPC_scripts/train_model_era5 ../HPC_scripts/train_model_era5_test.sh
#    ... will convert train_model_era5_template.sh to ../HPC_scripts/train_model_era5_test.sh
# **************** Description ****************
#
# **************** Auxiliary functions ****************


HOST_NAME=`hostname`
CURR_DIR_FULL=`pwd`
CURR_DIR="$(basename "$CURR_DIR_FULL")"
BASE_DIR="$(dirname "$CURR_DIR_FULL")"

### Some sanity checks ###
# ensure that the script is executed from the env_setup-subdirectory
if [[ "${CURR_DIR}" != "config_runscripts"  ]]; then
  echo "ERROR: Execute 'setup_runscript_templates.sh' from the config_runscripts-subdirectory only!"
  exit 1
fi
# check input arguments
if [[ "$#" -ne 1 ]]; then
  echo "ERROR: Pass path to base directory where the data of the worklfow steps should be saved."
  echo "Example: ./setup_runscript_templates.sh /p/project/deepacf/deeprain/video_prediction_shared_folder/"
  exit 1
else
  data_dir=$1
  base_data_dir="$(dirname "${data_dir}")"
  if [[ ! -d ${base_data_dir} ]]; then
    echo "ERROR: Top-level data directory ${base_data_dir} does not exist. Cannot create passed directory."
    exit 2
  fi
  if [[ ! -d ${data_dir} ]]; then
    mkdir ${data_dir}
    echo "Passed directory '${data_dir}' created successfully."
  fi
fi

echo "Start setting up templates under nonHPC_scripts/..."
for f in ../nonHPC_scripts/*template.sh; do
  echo ${f}
  sed -i "s|\(.*_dir=\).*|\1${data_dir}|g" ${f}
done
echo "Done!"

echo "Start setting up templates under HPC_scripts/"
for f in ../HPC_scripts/data*template.sh; do
  echo ${f}
  sed -i "s|\(.*_dir=\).*|\1${data_dir}|g" ${f}
done
# end
