#!/usr/bin/env bash

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


source ../video_prediction_tools/env_setup/modules_train.sh
python -m pytest test_era5_data.py

