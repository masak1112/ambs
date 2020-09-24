#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2020_09_24'
#
# **************** Description ****************
# Converts a given template workflow script (path has to be passed as first argument) to
# an executable workflow (Batch) script.
# Note, that the first argument has to be passed with "_template.sh" omitted!
# The second argument denotes the name of the virtual environment to be used.
# Additionally, -exp_id=[some_id] and -exp_dir=[some_dir] can be optionally passed as NON-POSITIONAL arguments.
# -exp_id allows to set an experimental identifier explicitly (default is -exp_id=exp1) while 
# -exp_dir allows setting manually the experimental directory where the preprocessed era5-data is stored.
# Note that the latter is helpful when the preprocessing step is skipped (since the data already exists!)
#
# Examples:
#    ./generate_workflow_scripts.sh ../HPC_scripts/generate_era5 venv_hdfml -exp_id=exp5
#    ... will convert generate_era5_template.sh to generate_era5_exp5.sh where
#    venv_hdfml is the virtual environment for operation.
#
#    ./generate_workflow_scripts.sh ../HPC_scripts/generate_era5 venv_hdfml -exp_id=exp5 -exp_dir=testdata
#    ... does the same as the previous example, but additionally extends source_dir=[...]/preprocessedData/
#    by testdata/
# **************** Description ****************
#
# **************** Auxilary function ****************
check_argin() {
# Handle input arguments and check if one of them holds -exp_id= 
# or -exp_dir= to emulate them as non-positional arguments
    for argin in "$@"; do
        if [[ $argin == *"-exp_id="* ]]; then
          exp_id=${argin#"-exp_id="}
        elif [[ $argin == *"-exp_dir="* ]]; then
          exp_dir=${argin#"-exp_dir="}
        fi
    done
}
# **************** Auxilary function ****************

HOST_NAME=`hostname`
BASE_DIR=`pwd`
WORKING_DIR="$(dirname "$BASE_DIR")"

### Some sanity checks ###
# check input arguments
if [[ "$#" -lt 2 ]]; then
  echo "ERROR: Pass path to workflow runscript (without '_template.sh') and pass name of virtual environment..."
  echo "Example: ./generate_workflow_scripts.sh ../HPC_scripts/DataExtraction venv_hdfml"
  exit 1
else
  curr_script=$1
  curr_script_loc="$(basename "$curr_script")"
  curr_venv=$2
  # check if any known non-positional argument is present...
  if [[ "$#" -gt 2 ]]; then
    check_argin ${@:3}
  fi
  #...and ensure that exp_id is always set
  if [[ -z "${exp_id}" ]]; then
    exp_id="exp1"
  fi
fi

# check existence of template script
if ! [[ -f ${curr_script}_template.sh ]]; then
  echo "WARNING: Could not find expected Batch script '${curr_script}_template.sh'."
  echo "Thus, no corresponding executable script is created!"
  if [[ ${curr_script} == *"template"* ||  ${curr_script} == *".sh"* ]]; then
    echo "ERROR: Omit '_template' and/or '.sh'  from Bash script argument."
    exit 2
  else
    exit 0              # still ok, i.e. only a WARNING is raised
  fi
fi

# Check existence of virtual environment (2nd argument)
if [[ ! -d ${WORKING_DIR}/${curr_venv} ]]; then
  echo "ERROR: Could not find directory of virtual environment under ${WORKING_DIR}/${curr_venv} "
  exit 3
fi

# Check if target script is unique
target_script=${curr_script}_${exp_id}.sh
if [[ -f ${target_script} ]]; then
  echo "ERROR: ${target_script} already exist."
  echo "Set explicitly a different experiment identifier."
  exit 4
fi

### Do the work ###
# create copy of template which is modified subsequently
cp ${curr_script}_template.sh ${target_script}
# remove template identifiers
num_lines=`awk '/Template identifier/{ print NR }' ${target_script}`
line_s=`echo ${num_lines} | cut -d' ' -f 1`
line_e=`echo ${num_lines} | cut -d' ' -f 2`
if [[ ${line_s} == "" || ${line_e} == "" ]]; then
  echo "ERROR: ${curr_script}_template.sh exists, but does not seem to be a valid template script."
  rm ${target_script}     # remove copy again
  exit 5
else
  sed -i "${line_s},${line_e}d" ${target_script}
fi

# set virtual environment to be used in Batch scripts
if [[ `grep "VIRT_ENV_NAME=" ${target_script}` ]]; then
  sed -i "s/VIRT_ENV_NAME=.*/VIRT_ENV_NAME=${curr_venv}/g" ${target_script}
fi

# also adapt name output- and error-files of submitted job with exp_id (if we are on Juwels or HDF-ML)
if [[ `grep "#SBATCH --output=" ${target_script}` ]]; then
  sed -i "s|#SBATCH --output=.*|#SBATCH --output=${curr_script_loc}_${exp_id}-out\.%j|g" ${target_script}
fi
if [[ `grep "#SBATCH --error=" ${target_script}` ]]; then
  sed -i "s|#SBATCH --error=.*|#SBATCH --error=${curr_script_loc}_${exp_id}-err\.%j|g" ${target_script}
fi

# set exp_id in (Batch) script if present
if [[ `grep "exp_id=" ${target_script}` ]]; then
  sed -i "s/exp_id=.*/exp_id=$exp_id/g" ${target_script}
fi

# set correct e-mail address in Batch scripts on Juwels and HDF-ML
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
  if ! [[ -z `command -v jutil` ]]; then
    USER_EMAIL=$(jutil user show -o json | grep email | cut -f2 -d':' | cut -f1 -d',' | cut -f2 -d'"')
  else
    USER_EMAIL=""
  fi
  sed -i "s/--mail-user=.*/--mail-user=$USER_EMAIL/g" ${target_script}
fi

# finally set experimental directory if exp_dir is present
if [[ ! -z "${exp_dir}" ]]; then
  if [[ `grep "/preprocessedData/$" ${target_script}` ]]; then    # the dollar-signs ensures that /preprocessedData/ is the suffix
                                                                  # i.e. this prevents us from modifying anything in DataPreprocess_[exp_id].sh 
                                                                  # where thsi is supposed to be done automatically
   sed -i "s|/preprocessedData/$|/preprocessedData/${exp_dir}/|g" ${target_script}
  else
   echo "WARNING: -exp_dir was passed, but no path with .../preprocessedData/ found in ${target_script}"
  fi
fi



