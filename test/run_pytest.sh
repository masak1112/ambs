#!#bin/bash
export PYTHONPATH=/p/project/deepacf/deeprain/gong1/${USER}/video_prediction_tools:$PYTHONPATH
# Name of virtual environment 
#VIRT_ENV_NAME="vp_new_structure"
VIRT_ENV_NAME="juwels_env"

if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../video_prediction_tools/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../video_prediction_tools/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      return
   fi
fi


source ../video_prediction_tools/env_setup/modules_data_extraction.sh
source ../video_prediction_tools/env_setup/modules_preprocess.sh
source ../video_prediction_tools/env_setup/modules_postprocess.sh
##Test data extraction

#ml Stages/2019a  GCC/8.3.0  ParaStationMPI/5.2.2-1 CDO/1.9.6
#python -m pytest  test_prepare_era5_data.py 
##Test for preprocess_step1
#python -m pytest  test_process_netCDF_v2.py
source ../video_prediction_tools/env_setup/modules_train.sh
#Test for process step2
#python -m pytest test_data_preprocess_step2.py
#python -m pytest test_era5_data.py
#Test for training
#First remove all the files in the test folder
#rm /p/project/deepacf/deeprain/video_prediction_shared_folder/models/test/* 
#python -m pytest test_train_model_era5.py
#python -m pytest test_vanilla_vae_model.py
python -m pytest test_visualize_postprocess.py
#python -m pytest test_meta_postprocess.py
