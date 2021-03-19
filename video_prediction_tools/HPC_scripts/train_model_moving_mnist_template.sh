#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_moving_mnist-out.%j
#SBATCH --error=train_moving_mnist-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment 
VIRT_ENV_NAME="my_venv"

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
destination_dir=${destination_dir}/${model}/"$(date +"%Y%m%dT%H%M")_"$USER""

# rund training

srun python ../scripts/train_dummy.py --input_dir  ${source_dir}/tfrecords/ --dataset moving_mnist  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/
