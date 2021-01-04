#!/usr/bin/env bash

# __author__ = Bing Gong, Michael Langguth
# __date__  = '2020_01_04'

# This script loads the required modules for the training workflow step of AMBS on Juwels, Juwels Booster and HDF-ML.
# Note that some other packages have to be installed into the virtual environment since not all Python-packages
# are available via the software stack (see create_env.sh and requirements.txt).

HOST_NAME=`hostname`

echo "Start loading modules on ${HOST_NAME}..."
echo "modules_train.sh is subject to: "
echo "* preprocess_data_era5_step2_<exp_id>.sh"
echo "* train_model_era5_<exp_id>.sh"

module purge
module use $OTHERSTAGES
if [[ "${HOST_NAME}" == jwlogin21* || "${HOST_NAME}" == jwlogin22* || "${HOST_NAME}" == jwlogin23* ]]; then
  ml Stages/2020
  ml UCX/1.8.1
  ml GCC/9.3.0
  ml OpenMPI/4.1.0rc1
else
  ml Stages/2019a
  ml GCC/8.3.0
  ml MVAPICH2/2.3.3-GDR
  ml mpi4py/3.0.1-Python-3.6.8
  # serialized version of HDF5 is used since only this version is compatible with TensorFlow/1.13.1-GPU-Python-3.6.8
  ml h5py/2.9.0-serial-Python-3.6.8
  ml TensorFlow/1.13.1-GPU-Python-3.6.8
  ml Horovod/0.16.2-GPU-Python-3.6.8
  ml cuDNN/7.5.1.10-CUDA-10.1.105
  ml netcdf4-python/1.5.0.1-Python-3.6.8
fi


