#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# auxiliary variables
WORK_DIR="$(pwd)"
BASE_DIR=$(dirname "$WORK_DIR")
# Name of virtual environment
VIRT_ENV_NAME=venv_test

# declare directory-variables which will be modified by config_runscript.py
source_dir=/my/path/to/pkl/files/
destination_dir=/my/path/to/tfrecords/files

sequence_length=20
sequences_per_file=10

#the number of the nodes should be the number of processed folder (month) plus 1
n_nodes=3

# !!! ADAPAT DEPENDING ON USAGE OF CONTAINER !!!
# For container usage, comment in the follwoing lines
# Name of container image (must be available in working directory)
CONTAINER_IMG="${WORK_DIR}/tensorflow_21.09-tf1-py3.sif"
WRAPPER="${BASE_DIR}/env_setup/wrapper_container.sh"

# sanity checks
if [[ ! -f ${CONTAINER_IMG} ]]; then
  echo "ERROR: Cannot find required TF1.15 container image '${CONTAINER_IMG}'."
  exit 1
fi

if [[ ! -f ${WRAPPER} ]]; then
  echo "ERROR: Cannot find wrapper-script '${WRAPPER}' for TF1.15 container image."
  exit 1
fi

mpirun -n singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} \
python3 ../main_scripts/main_preprocess_data_step2.py -source_dir ${source_dir} -dest_dir ${destination_dir} \
-sequence_length ${sequence_length} -sequences_per_file ${sequences_per_file}

# WITHOUT container usage, comment in the follwoing lines (and uncomment the lines above)
# Activate virtual environment if needed (and possible)
#if [ -z ${VIRTUAL_ENV} ]; then
#   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
#      echo "Activating virtual environment..."
#      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
#   else
#      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
#      exit 1
#   fi
#fi
#
# Run preprocessing step 2
#mpirun -n ${n_nodes}  python3 ../main_scripts/main_preprocess_data_step2.py -source_dir ${source_dir} -dest_dir ${destination_dir} \
#     -sequence_length ${sequence_length} -sequences_per_file ${sequences_per_file}





