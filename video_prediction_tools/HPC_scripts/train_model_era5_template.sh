#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_era5-out.%j
#SBATCH --error=train_era5-err.%j
#SBATCH --time=20:00:00
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

# declare directory-variables which will be modified by config_runscript.py
source_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/
destination_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/models/

# valid identifiers for model-argument are: convLSTM, savp, mcnet and vae
# the destination_dir_full cannot end up with "/", this will cause to save all the checkpoints issue in the results_dir
model=convLSTM
datasplit_dict=../data_split/cv_test.json
model_hparams=${destination_dir}/model_hparams.json
dataset=era5

#If you train savp, Please uncomment the following CUDA configuration
#CUDA_VISIBLE_DEVICES=1

# run training
srun python ../main_scripts/main_train_models.py --input_dir  ${source_dir} --datasplit_dict ${datasplit_dict} \
 --dataset ${dataset}  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir} --checkpoint_dir  ${destination_dir}
 
