#!/usr/bin/env bash

# __author__ = Bing Gong, Michael Langguth
# __date__  = '2021_01_15'

# This script loads the required modules for the training workflow step of AMBS on Juwels, Juwels Booster and HDF-ML.
# Note that some other packages have to be installed into the virtual environment since not all Python-packages
# are available via the software stack (see create_env.sh and requirements.txt).

HOST_NAME=`hostname`

echo "Start loading modules on ${HOST_NAME}..."
echo "modules_train.sh is subject to: "
echo "* preprocess_data_era5_step2_<exp_id>.sh"
echo "* train_model_era5_[booster_]<exp_id>.sh"
echo "* visualize_postprocess_era5_<exp_id>.sh"

module use $OTHERSTAGES
if [[ "${HOST_NAME}" == jwlogin2[1-4]* || "${HOST_NAME}" == jwb* ]]; then
  ml Stages/2020
  ml UCX/1.8.1
  ml GCC/9.3.0
  ml OpenMPI/4.1.0rc1
else
  ml Stages/2019a
  ml GCC/8.3.0
  ml ParaStationMPI/5.4.4-1
  ml mpi4py/3.0.1-Python-3.6.8
  ml h5py/2.9.0-serial-Python-3.6.8
  ml TensorFlow/1.13.1-GPU-Python-3.6.8
  ml cuDNN/7.5.1.10-CUDA-10.1.105
  ml SciPy-Stack/2019a-Python-3.6.8
  ml scikit/2019a-Python-3.6.8
  ml netcdf4-python/1.5.0.1-Python-3.6.8
  # Horovod is excluded as long as parallelization does not work properly
  # Note: Horovod/0.16.2 requires MVAPICH2 which is incomaptible with netcdf4-python
  #ml MVAPICH2/2.3.3-GDR               # 
  #ml Horovod/0.16.2-GPU-Python-3.6.8
fi

# clean up if triggered via script argument
if [[ $1 == purge ]]; then
  echo "Purge all modules after loading them..."
  module --force purge
fi  
