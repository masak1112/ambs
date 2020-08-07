#!/usr/bin/env bash

# __author__ = Bing Gong, Michael Langguth
# __date__  = '2020_06_26'

# This script loads the required modules for ambs on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements.txt).

HOST_NAME=`hostname`

echo "Start loading modules on ${HOST_NAME}..."

module purge
module use $OTHERSTAGES
module load Stages/2019a
module load GCC/8.3.0
module load GCCcore/.8.3.0
module load ParaStationMPI/5.2.2-1
module load mpi4py/3.0.1-Python-3.6.8
# serialized version of HDF5 is used since only this version is compatible with TensorFlow/1.13.1-GPU-Python-3.6.8
module load h5py/2.9.0-serial-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load cuDNN/7.5.1.10-CUDA-10.1.105
module load netcdf4-python/1.5.0.1-Python-3.6.8

