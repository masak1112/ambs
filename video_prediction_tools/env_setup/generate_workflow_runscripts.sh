#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2020_09_29'
#
# **************** Description ****************
# Converts a given template workflow script (path/name has to be passed as first argument) to
# an executable workflow (Batch) script.
# Note, that the first argument has to be passed with "_template.sh" omitted!
# The second argument denotes the name of the virtual environment to be used.
# Additionally, -exp_id=[some_id] and -exp_dir=[some_dir] can be optionally passed as NON-POSITIONAL arguments.
# -exp_id allows to set an experimental identifier explicitly (default is -exp_id=exp1) while 
# -exp_dir allows setting manually the experimental directory.
# Note, that the latter is done during the preprocessing step in an end-to-end workflow.
# However, if the preprocessing step can be skipped (i.e. preprocessed data already exists),
# one may wish to set the experimental directory explicitly
#
# Examples:
#    ./generate_workflow_scripts.sh ../HPC_scripts/generate_era5 venv_hdfml -exp_id=exp5
#    ... will convert generate_era5_template.sh to generate_era5_exp5.sh where
#    venv_hdfml is the virtual environment for operation.
#
#    ./generate_workflow_scripts.sh ../HPC_scripts/generate_era5 venv_hdfml -exp_id=exp5 -exp_dir=testdata
#    ... does the same as the previous example, but additionally extends source_dir=[...]/preprocessedData/,
#    checkpoint_dir=[...]/models/ and results_dir=[...]/results/ by testdata/
# **************** Description ****************
#
# **************** Auxilary functions ****************
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
   echo "PErform extension on path '${prefix}/' in runscript '${target_script}'"
   sed -i "s|/${prefix}/$|/${prefix}/${extension}/|g" ${target_script}
   status=1
  fi
}
# **************** Auxilary functions ****************

HOST_NAME=`hostname`
BASE_DIR=`pwd`
WORKING_DIR="$(dirname "$BASE_DIR")"
EXE_DIR="$(basename "$BASE_DIR")"

### Some sanity checks ###
# ensure that the script is executed from the env_setup-subdirectory
if [[ "${EXE_DIR}" != "env_setup"  ]]; then
  echo "ERROR: Execute 'generate_workflow_scripts.sh' from the env_setup-subdirectory only!"
  exit 1
fi
# check input arguments
if [[ "$#" -lt 2 ]]; then
  echo "ERROR: Pass path to workflow runscript (without '_template.sh') and pass name of virtual environment..."
  echo "Example: ./generate_workflow_scripts.sh ../HPC_scripts/DataExtraction venv_hdfml"
  exit 1
else
  curr_script=$1
  curr_script_loc="$(basename "$curr_script")"
  curr_venv=$2                          #
  # check if any known non-positional argument is present...
  if [[ "$#" -gt 2 ]]; then
    check_argin ${@:3}
    if [[ ! -z "${exp_dir_ext}" ]] && [[ -z "${exp_dir}" ]]; then
      echo "WARNING: -exp_dir_ext is passed without passing -ext_dir and thus has no effect!"
    fi
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
else 
  echo "Convert ${curr_script}_template.sh to executable runscript"
  echo "The executable runscript is saved under ${target_script}" 
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
  sed -i 's/VIRT_ENV_NAME=.*/VIRT_ENV_NAME="'${curr_venv}'"/g' ${target_script}
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

# set model if model was passed as optional argument
if [[ ! -z "${model}" ]]; then
  sed -i "s/model=.*/model=${model}/g" ${target_script}
fi

# finally set experimental directory if exp_dir is present
if [[ ! -z "${exp_dir}" ]]; then
  if [[ ! -z "${exp_dir_ext}" ]]; then
    status=0                      # status to check if exp_dir_ext is added to the runscript at hand
                                  # -> will be set to one by extend_path if modifictaion takes place
    extend_path models ${exp_dir_ext}/
    extend_path results ${exp_dir_ext}

    if [[ ${status} == 0 ]]; then
      echo "WARNING: -exp_dir_ext has been passed, but no addition to any path in runscript at hand done..."
    fi
  fi
  status=0                        # status to check if exp_dir is added to the runscript at hand
                                  # -> will be set to one by add_exp_dir if modifictaion takes place
  extend_path preprocessedData ${exp_dir}

  if [[ ${status} == 0 ]]; then
    echo "WARNING: -exp_dir has been passed, but no addition to any path in runscript at hand done..."
  fi
fi



