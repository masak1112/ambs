#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_era5_step2-out.%j
#SBATCH --error=DataPreprocess_era5_step2-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

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
destination_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/
# further settings
datasplit_dir=../data_split/cv_test.json
model=convLSTM
hparams_dict_config=../hparams/era5/${model}/model_hparams.json
sequences_per_file=10
sequence_length=20

# run preprocessing (step 2 where Tf-records are generated)
srun python ../main_scripts/main_preprocess_data_step2.py -input_dir ${source_dir} -output_dir ${destination_dir}  -datasplit_config ${datasplit_dir}  -hparams_dict_config ${hparams_dict_config} -sequences_per_file ${sequences_per_file}

