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

# Declare input parameters
root_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/
analysis_config=video_prediction_tools/meta_postprocess_config/meta_config.json
metric=mse
exp_id=test
enable_skill_scores=True

srun python ../main_scripts/main_meta_postprocess.py  --root_dir ${root_dir} --analysis_config ${analysis_config} \
                                                       --metric ${metric} --exp_id ${exp_id} --enable_skill_scores ${enable_skill_scores}
