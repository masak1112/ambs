#!#bin/bash
export PYTHONPATH=$WORKDIR/ambs/video_prediction_tools:$PYTHONPATH
# Name of virtual environment 
VIRT_ENV_NAME="vp_new_structure"


if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../video_prediction_tools/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../video_prediction_tools/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi



source ../video_prediction_tools/env_setup/modules_preprocess.sh
python -m pytest  test_process_netCDF_v2.py

