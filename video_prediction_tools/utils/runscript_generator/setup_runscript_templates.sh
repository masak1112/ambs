#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2021_02_05'
#
# **************** Description ****************
# Sets the base directory to the template runscripts under which all the data will be stored,
# i.e. where the AMBS-directory tree will be set up.
# If no argument is passed, the default defined by 'base_data_dir_default' is set.
#
# Example:
#    ./setup_runscript_templates.sh [<my_path>]
# **************** Description ****************
#
# **************** Auxiliary functions ****************

# default value for base directory
base_data_dir_default=/p/project/deepacf/deeprain/video_prediction_shared_folder/
# base_data_dir_default=/p/scratch/deepacf/ji4/
# some further directory paths
CURR_DIR_FULL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"   # retrieves the location of this script
BASE_DIR="$(dirname "$(dirname "${CURR_DIR_FULL}")")"
USER=$USER

### Some sanity checks ###
# check/handle input arguments
if [[ "$#" -lt 1 ]]; then
  data_dir=${base_data_dir_default}
  echo "No base directory passed. Thus, the default path ${base_data_dir_default} will be applied."
  echo "In order to set it pass the directory path as a first argument."
  echo "Example: ./setup_runscript_templates.sh /my/desired/path/"
elif [[ "$#" -ge 2 ]]; then
  echo "ERROR: Too many arguments provided. Cannot continue..."
  exit 1
else
  data_dir=$1
  base_data_dir="$(dirname "${data_dir}")"
  if [[ ! -d ${base_data_dir} ]]; then
    echo "ERROR: Top-level data directory ${base_data_dir} does not exist. Cannot create passed directory."
    exit 2
  fi
  if [[ ! -d ${data_dir} ]]; then
    mkdir "${data_dir}"
    echo "Passed directory '${data_dir}' created successfully."
  fi
fi

echo "Start setting up templates under nonHPC_scripts/..."
for f in "${BASE_DIR}"/nonHPC_scripts/*template.sh; do
  echo "Setting up ${f}..."
  fnew=${f%%.*}_${USER}.sh
  cp "${f}" "${fnew}"
  sed -i "s|\(.*_dir=\).*|\1${data_dir}|g" "${fnew}"
done
echo "Done!"

echo "Start setting up templates under HPC_scripts/"
for f in "${BASE_DIR}"/HPC_scripts/*template.sh; do
  echo "Setting up ${f}..."
  fnew=${f%%.*}_${USER}.sh
  cp "${f}" "${fnew}"
  sed -i "s|\(.*_dir=\).*|\1${data_dir}|g" "${fnew}"
done
echo "Done!"
# end
