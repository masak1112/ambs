#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train_model_era5_container-out.%j
#SBATCH --error=train_model_era5_container-err.%j
#SBATCH --time=24:00:00
##SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

### Two nodes, 8 GPUs
##SBATCH --nodes=2
##SBATCH --ntasks=8
##SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:4
## Also take care for the job submission with srun below!!!


WORK_DIR=`pwd`
BASE_DIR=$(dirname "$WORK_DIR")
# Name of virtual environment
VIRT_ENV_NAME="my_venv"
# Name of container image (must be available in working directory)
CONTAINER_IMG="${WORK_DIR}/tensorflow_21.09-tf1-py3.sif"

# purge modules to avoid conflicts between container and host settings
module purge

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/my/path/to/tfrecords/files
destination_dir=/my/model/output/path

# valid identifiers for model-argument are: convLSTM, savp, mcnet and vae
model=convLSTM
datasplit_dict=${destination_dir}/data_split.json
model_hparams=${destination_dir}/model_hparams.json

# run training in container
export CUDA_VISIBLE_DEVICES=0,1,2,3
## One node, single GPU 
srun --mpi=pspmix --cpu-bind=none \
singularity exec --nv ${CONTAINER_IMG} ./wrapper_container.sh ${VIRT_ENV_NAME} python3 ${BASE_DIR}/main_scripts/main_train_models.py \
	--input_dir ${source_dir}  --datasplit_dict ${datasplit_dict}  --dataset era5 --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/

## Two nodes, 8 GPUs 
#srun -N 2 -n 8 --ntasks-per-node 4 singularity exec --nv ${CONTAINER_IMG} ./wrapper_container.sh ${VIRT_ENV_NAME} python3 ${BASE_DIR}/main_scripts/main_train_models.py \
#	--input_dir ${source_dir} --datasplit_dict ${datasplit_dict} --dataset era5 --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/
