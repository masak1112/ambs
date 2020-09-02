#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
<<<<<<< Updated upstream
#SBATCH --output=train_moving_mnist-out.%j
#SBATCH --error=train_moving_mnist-err.%j
=======
#SBATCH --output=train_era5-out.%j
#SBATCH --error=train_era5-err.%j
>>>>>>> Stashed changes
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
destination_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/models/moving_mnist

# for choosing the model, convLSTM,savp, mcnet,vae
model=convLSTM
dataset=moving_mnist
model_hparams=../hparams/${dataset}/${model}/model_hparams.json

# rund training

srun python ../scripts/train_dummy.py --input_dir  ${source_dir}/tfrecords/ --dataset moving_mnist  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/${model}_bing_20200902/ 

