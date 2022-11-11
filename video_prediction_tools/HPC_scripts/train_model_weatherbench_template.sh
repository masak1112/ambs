#!/bin/bash -x
#SBATCH --account=<your_project>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train_model_era5-out.%j
#SBATCH --error=train_model_era5-err.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=some_partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# auxiliary variables
WORK_DIR="$(pwd)"
BASE_DIR=$(dirname "$WORK_DIR")
# Name of virtual environment
VIRT_ENV_NAME="my_venv"
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

# clean-up modules to avoid conflicts between host and container settings
module purge

# declare directory-variables which will be modified by generate_runscript.py
source_dir=/my/path/to/tfrecords/files
destination_dir=/my/model/output/path

# valid identifiers for model-argument are: convLSTM, savp, mcnet and vae
model=convLSTM
datasplit_dict=${destination_dir}/data_split.json
model_hparams=${destination_dir}/model_hparams.json

# run training in container
export CUDA_VISIBLE_DEVICES=0
## One node, single GPU 
srun --mpi=pspmix --cpu-bind=none \
     singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} \
     python3 "${BASE_DIR}"/main_scripts/main_train_models.py --input_dir ${source_dir} --datasplit_dict ${datasplit_dict} \
     --dataset weatherbench --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/

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
# Loading modules
#module purge
#source ../env_setup/modules_train.sh
#export CUDA_VISIBLE_DEVICES=0
#
# srun python3 "${BASE_DIR}"/main_scripts/main_train_models.py --input_dir ${source_dir} --datasplit_dict ${datasplit_dict} \
#     --dataset era5 --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/