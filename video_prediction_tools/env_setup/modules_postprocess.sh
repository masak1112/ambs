#!/usr/bin/env bash

# __author__ = Bing Gong, Michael Langguth
# __date__  = '2021_01_04'

# This script loads the required modules for the postprocessing workflow step of AMBS on Juwels and HDF-ML.
# Note that some other packages have to be installed into the virtual environment since not all Python-packages
# are available via the software stack (see create_env.sh and requirements.txt).

HOST_NAME=`hostname`

echo "Start loading modules on ${HOST_NAME}..."
echo "modules_postprocess.sh is subject to: "
echo "* visualize_postprocess_era5_<exp_id>.sh"

module purge
module use $OTHERSTAGES
ml Stages/2019a
ml GCC/8.3.0
ml GCCcore/.8.3.0
ml ParaStationMPI/5.2.2-1
ml mpi4py/3.0.1-Python-3.6.8
# serialized version of HDF5 is used since only this version is compatible with TensorFlow/1.13.1-GPU-Python-3.6.8
ml h5py/2.9.0-serial-Python-3.6.8
ml TensorFlow/1.13.1-GPU-Python-3.6.8
ml cuDNN/7.5.1.10-CUDA-10.1.105
ml SciPy-Stack/2019a-Python-3.6.8
ml scikit/2019a-Python-3.6.8
ml netcdf4-python/1.5.0.1-Python-3.6.8

# clean up if triggered via script argument
if [[ $1 == purge ]]; then
  echo "Purge all modules after loading them..."
  module --force purge
fi  
