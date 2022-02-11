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
pip install --no-cache-dir -r ../env_setup/requirements.txt

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


# declare directory-variables which will be modified by generate_runscript.py
# Note: source_dir is only needed for retrieving the base-directory
source_dir=/my/source/dir/
checkpoint_dir=/my/trained/model/dir
results_dir=/my/results/dir
lquick=""

# run postprocessing/generation of model results including evaluation metrics
export CUDA_VISIBLE_DEVICES=0
## One node, single GPU
srun --mpi=pspmix --cpu-bind=none \
     singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} \
     python3 ../main_scripts/main_visualize_postprocess.py --checkpoint  ${checkpoint_dir} --mode test  \
                                                           --results_dir ${results_dir} --batch_size 4 \
                                                           --num_stochastic_samples 1 ${lquick} \
                                                           > postprocess_era5-out_all."${SLURM_JOB_ID}"

