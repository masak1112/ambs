#!#bin/bash

# Name of virtual environment
VIRT_ENV_NAME="venv2_hdfml"

if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../video_prediction_tools/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../video_prediction_tools/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      return
   fi
fi


## if you do test on data extraction and data preprocess you need to source the modules_preprocess.sh
#source ../video_prediction_tools/env_setup/modules_preprocess.sh
##Test data extraction
#python -m pytest  test_prepare_era5_data.py 
##Test for preprocess_step1
#python -m pytest  test_process_netCDF_v2.py
#source ../video_prediction_tools/env_setup/modules_preprocess+extract.sh
source ../video_prediction_tools/env_setup/modules_train.sh
##Test for preprocess moving mnist
#python -m pytest test_prepare_moving_mnist_data.py
#python -m pytest test_train_moving_mnist_data.py 
#Test for process step2
#python -m pytest test_data_preprocess_step2.py
#python -m pytest test_era5_data.py
#Test for training
#First remove all the files in the test folder
#rm /p/project/deepacf/deeprain/video_prediction_shared_folder/models/test/* 
#python -m pytest test_train_model_era5.py
#python -m pytest test_vanilla_vae_model.py
python -m pytest test_gzprcp_data.py
#python -m pytest test_meta_postprocess.py
