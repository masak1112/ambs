#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2021_02_02'
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
# ML 2021-02-02: Not in use, but retained since they are quite helpful
check_argin() {
# Handle input arguments and check if one of them holds -exp_id= 
# or -exp_dir= to emulate them as non-positional arguments
    for argin in "$@"; do
        if [[ $argin == *"-exp_id="* ]]; then
          exp_id=${argin#"-exp_id="}
        elif [[ $argin == *"-exp_dir="* ]]; then
          exp_dir=${argin#"-exp_dir="}
        elif [[ $argin == *"-exp_dir_ext"* ]]; then
          exp_dir_ext=${argin#"-exp_dir_ext="}
        elif [[ $argin == *"-model"* ]]; then
          model=${argin#"-model="}
        fi
    done
}

extend_path() {
# Add <extension> to paths in <target_script> which end with /<prefix>/
  prefix=$1
  extension=$2
  if [[ `grep "/${prefix}/$" ${target_script}` ]]; then
   echo "Perform extension on path '${prefix}/' in runscript '${target_script}'"
   sed -i "s|/${prefix}/$|/${prefix}/${extension}/|g" ${target_script}
   status=1
  fi
}
# **************** Auxiliary functions ****************

HOST_NAME=`hostname`

SCR_NAME="convert_runscript.sh"

suffix=".sh"
### Some sanity checks ###
# check input arguments
if [[ "$#" -ne 2 ]]; then
  echo "%${SCR_NAME}: ERROR: Pass path to workflow runscript as well as name of target file"
  echo "%${SCR_NAME}: Example: ./convert_runscript.sh ../HPC_scripts/DataExtraction_template.sh ../HPC_scripts/DataExtraction_test.sh"
  exit 1
else
  curr_script=$1
  target_script=$2
  log_str=${target_script%"$suffix"}
  echo ${target_snippet}
fi

# check existence of template script
if ! [[ -f ${curr_script} ]]; then
  echo "%${SCR_NAME}: EXIT: Could not find expected Batch script '${curr_script}'."
  echo "%${SCR_NAME}: Thus, no corresponding executable script is created!"
  exit 2
fi

# Check if target script is unique
echo "%${SCR_NAME}: Convert ${curr_script} to executable runscript"
echo "%${SCR_NAME}: The executable runscript is saved under ${target_script}"

### Do the work ###
# create copy of template which is modified subsequently
cp ${curr_script} ${target_script}
# remove template identifiers
num_lines=`awk '/Template identifier/{ print NR }' ${target_script}`
line_s=`echo ${num_lines} | cut -d' ' -f 1`
line_e=`echo ${num_lines} | cut -d' ' -f 2`
if [[ ${line_s} == "" || ${line_e} == "" ]]; then
  echo "%${SCR_NAME} ERROR: ${curr_script} exists, but does not seem to be a valid template script."
  rm ${target_script}     # remove copy again
  exit 3
else
  sed -i "${line_s},${line_e}d" ${target_script}
fi
# also adapt name output- and error-files of submitted job with exp_id (if we are on Juwels or HDF-ML)
if [[ `grep "#SBATCH --output=" ${target_script}` ]]; then
  sed -i "s|#SBATCH --output=.*|#SBATCH --output=${log_str}-out\.%j|g" ${target_script}
fi
if [[ `grep "#SBATCH --error=" ${target_script}` ]]; then
  sed -i "s|#SBATCH --error=.*|#SBATCH --error=${log_str}-err\.%j|g" ${target_script}
fi
# set correct e-mail address in Batch scripts on Juwels and HDF-ML
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *juwels* ]]; then
  if ! [[ -z `command -v jutil` ]]; then
    USER_EMAIL=$(jutil user show -o json | grep email | cut -f2 -d':' | cut -f1 -d',' | cut -f2 -d'"')
  else
    USER_EMAIL=""
  fi
  sed -i "s/--mail-user=.*/--mail-user=$USER_EMAIL/g" ${target_script}
fi
# end
