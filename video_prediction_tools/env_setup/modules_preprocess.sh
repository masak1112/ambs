#!/usr/bin/env bash

# __author__ = Bing Gong, Michael Langguth
# __date__  = '2020_06_26'

# This script loads the required modules for ambs on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements.txt).

HOST_NAME=`hostname`

echo "Start loading modules on ${HOST_NAME} required for preprocessing..."
echo "modules_preprocess.sh is subject to: "
echo "* data_extraction_era5.sh"
echo "* preprocess_data_era5_step1.sh"

module purge
module use $OTHERSTAGES
ml Stages/2019a
ml GCC/8.3.0
ml ParaStationMPI/5.2.2-1
ml mpi4py/3.0.1-Python-3.6.8
# serialized version is not available on HFML
# see https://gitlab.version.fz-juelich.de/haf/Wiki/-/wikis/HDF-ML%20System
if [[ "${HOST_NAME}" == hdfml* ]]; then
  ml h5py/2.9.0-serial-Python-3.6.8
elif [[ "${HOST_NAME}" == juwels* ]]; then
  ml h5py/2.9.0-Python-3.6.8
fi
ml SciPy-Stack/2019a-Python-3.6.8
ml scikit/2019a-Python-3.6.8
ml netcdf4-python/1.5.0.1-Python-3.6.8

# clean up if triggered via script argument
if [[ $1 == purge ]]; then
  echo "Purge all modules after loading them..."
  module --force purge
fi  
