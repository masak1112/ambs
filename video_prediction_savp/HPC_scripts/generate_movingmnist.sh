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
#SBATCH --mail-user=s.stadtler@fz-juelich.de
##jutil env activate -p cjjsc42

# Name of virtual environment 
VIRT_ENV_NAME="vp"

# Loading mouldes
source ../env_setup/modules_train.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../${VIRT_ENV_NAME}/bin/activate
   else 
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/moving_mnist
checkpoint_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/models/moving_mnist
results_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/results/moving_mnist
# name of model
model=convLSTM

# run postprocessing/generation of model results including evaluation metrics
srun python -u ../scripts/generate_movingmnist.py \
--input_dir ${source_dir}/ --dataset_hparams sequence_length=20 --checkpoint  ${checkpoint_dir}/${model} \
--mode test --model ${model} --results_dir ${results_dir}/${model} --batch_size 2 --dataset era5   > generate_era5-out.out

#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
