#!/usr/bin/env bash
# **************** Description ****************
# Converts given template workflow script (path has to be passed as first argument) to
# an executable workflow (Batch) script.
# Note, that this first argument has to be passed with "_template.sh" omitted!
# A second argument can be passed to set an experiment identifier whose default is exp1.
# Note, that the second argument can be omitted only if there are no existing (Batch) scritps
# carrying this identifier which is added as a suffix.
# Example:
#    ./generate_workflow_scripts.sh ../HPC_scripts/generate exp5
#    ... will convert generate_template.sh to generate_exp5.sh
# **************** Description ****************
#

HOST_NAME=`hostname`

### some sanity checks ###
# check input arguments
if [[ "$#" -lt 1 ]]; then
  echo "ERROR: Pass path to workflow runscript (without '_template.sh') to be generated..."
  exit 1
else
  curr_script=$1
  if [[ "$#" -gt 1 ]]; then
    exp_id=$2
  else
    exp_id="exp1"
  fi
fi

# check existence of template script
if ! [[ -f ${curr_script}_template.sh ]]; then
  echo "WARNING: Could not find expected Batch script '${curr_script}_template.sh'."
  echo "Thus, no corresponding executable script is created!"
  if [[ ${curr_script} == *"template"* ||  ${curr_script} == *".sh"* ]]; then
    echo "ERROR: Omit '_template' and/or '.sh'  from Bash script argument."
    exit 1
  else
    exit 0              # still ok, i.e. only a WARNING is raised
  fi
fi
# check if target script is unique
target_script=${curr_script}_${exp_id}.sh
if [[ -f ${target_script} ]]; then
  echo "ERROR: ${target_script} already exist."
  echo "Set explicitly a different experiment identifier."
  exit 1
fi
### do the work ###
# create copy of template which is modified subsequently
cp ${curr_script}_template.sh ${target_script}
# remove template identifiers
num_lines=`awk '/Template identifier/{ print NR }' ${target_script}`
line_s=`echo ${num_lines} | cut -d' ' -f 1`
line_e=`echo ${num_lines} | cut -d' ' -f 2`
if [[ ${line_s} == "" || ${line_e} == "" ]]; then
  echo "ERROR: ${curr_script}_template.sh exists, but does not seem to be a valid template script."
  rm ${target_script}     # remove copy again
  exit 1
else
  sed -i "${line_s},${line_e}d" ${target_script}
fi
# set exp_id in (Batch) script if present
if [[ `grep "exp_id=" ${target_script}` ]]; then
  sed -i "s/exp_id=.*/exp_id=$exp_id/g" ${target_script}
fi

# set correct e-mail address in Batch scripts on Juwels and HDF-ML
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == juwels* ]]; then
  if [ command -v jutil ]; then
    USER_EMAIL=$(jutil user show -o json | grep email | cut -f2 -d':' | cut -f1 -d',' | cut -f2 -d'"')
  else
    USER_EMAIL=""
  fi
  sed -i "s/--mail-user=.*/--mail-user=$USER_EMAIL/g" ${target_script}
fi


