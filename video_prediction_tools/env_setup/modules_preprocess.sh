#!/usr/bin/env bash

# __author__ = Bing Gong, Michael Langguth
# __date__  = '2020_06_26'

# This script loads the required modules for ambs on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements.txt).

HOST_NAME=`hostname`

echo "Start loading modules on ${HOST_NAME} required for preprocessing..."
echo "modules_preprocess.sh is subject to: "
echo "* data_extraction_era5_<exp_id>.sh"
echo "* preprocess_data_era5_step1_<exp_id>.sh"

module purge
module use $OTHERSTAGES
module load Stages/2019a
module load GCC/8.3.0
module load ParaStationMPI/5.2.2-1
module load mpi4py/3.0.1-Python-3.6.8
# serialized version is not available on HFML
# see https://gitlab.version.fz-juelich.de/haf/Wiki/-/wikis/HDF-ML%20System
if [[ "${HOST_NAME}" == hdfml* ]]; then
  module load h5py/2.9.0-serial-Python-3.6.8
elif [[ "${HOST_NAME}" == juwels* ]]; then
  module load h5py/2.9.0-Python-3.6.8
fi
module load netcdf4-python/1.5.0.1-Python-3.6.8

