#!#bin/bash
export PYTHONPATH=$WORKDIR/ambs/video_prediction_tools:$PYTHONPATH
source ../video_prediction_tools/env_setup/modules_preprocess.sh
python -m pytest  test_process_netCDF_v2.py

