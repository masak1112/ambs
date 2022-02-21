#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=generate_era5-out.%j
#SBATCH --error=generate_era5-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# auxiliary variables
WORK_DIR="$(pwd)"
BASE_DIR=$(dirname "$WORK_DIR")
# Name of virtual environment
VIRT_ENV_NAME="my_venv"
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
source_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/moving_mnist
checkpoint_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/models/moving_mnist
results_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/results/moving_mnist
# name of model
model=convLSTM

# run postprocessing/generation of model results including evaluation metrics
export CUDA_VISIBLE_DEVICES=0
## One node, single GPU
srun --mpi=pspmix --cpu-bind=none \
     singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} \
     python3 ../scripts/generate_movingmnist.py --input_dir ${source_dir}/ --dataset_hparams sequence_length=20 \
     --checkpoint  ${checkpoint_dir}/${model} --mode test --model ${model} --results_dir ${results_dir}/${model} \
     --batch_size 2 --dataset era5 > generate_era5-out."${SLURM_JOB_ID}"

