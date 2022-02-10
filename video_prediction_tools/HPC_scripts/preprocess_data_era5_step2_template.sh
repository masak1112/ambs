#!/bin/bash -x
## Controlling Batch-job: Need input
#SBATCH --account=<Project name>
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=Data_Preprocess_step1_era5-out.%j
#SBATCH --error=Data_Preprocess_step1era5-err.%j
#SBATCH --time=04:20:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

##Load basic Python module: Need input
#module load Python


##Create and activate a virtual environment: Need input
#VENV_NAME=<my_venv>
#Python -m venv ../virtual_envs/${VENV_NAME}
#source ../virtual_envs/${VENV_NAME}/bin/activate

## Install required packages
# set PYTHONPATH...
WORKING_DIR="$(pwd)"
BASE_DIR=="$(WORKING_DIR "$dir")"
export PYTHONPATH=${BASE_DIR}/virtual_envs/${VENV_NAME}/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/model_modules:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH
# ... install requirements
pip install --no-cache-dir -r ../env_setup/requirements_nonJSC.txt

# Name of virtual environment
VENV_NAME=venv_hdfml
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

# clean-up modules to avoid conflicts between host and container settings
module purge

# declare directory-variables which will be modified by config_runscript.py
source_dir=/my/path/to/pkl/files/
destination_dir=/my/path/to/tfrecords/files

sequence_length=24
sequences_per_file=10
# run Preprocessing (step 2 where Tf-records are generated)
# run postprocessing/generation of model results including evaluation metrics
export CUDA_VISIBLE_DEVICES=0
## One node, single GPU
srun --mpi=pspmix --cpu-bind=none \
     singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VENV_NAME} \
     python3 ../main_scripts/main_preprocess_data_step2.py -source_dir ${source_dir} -dest_dir ${destination_dir} \
     -sequence_length ${sequence_length} -sequences_per_file ${sequences_per_file}

