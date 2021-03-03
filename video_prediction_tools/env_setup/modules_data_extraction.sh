#!/usr/bin/env bash

# __author__ = Bing Gong, Michael Langguth
# __date__  = '2020_06_26'

# This script loads the required modules for ambs on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements.txt).

HOST_NAME=`hostname`

echo "Start loading modules on ${HOST_NAME} required for era5 data extraction..."
echo "modules_data_etraction.sh is subject to: "
echo "* data_extraction_era5_<exp_id>.sh"

module purge
# serialized version is not available on HFML
# see https://gitlab.version.fz-juelich.de/haf/Wiki/-/wikis/HDF-ML%20System
if [[ "${HOST_NAME}" == hdfml* ]]; then
 module use $OTHERSTAGES
 ml Stages/2019a
 ml GCC/8.3.0
 ml ParaStationMPI/5.2.2-1
 ml CDO/1.9.6
 ml mpi4py/3.0.1-Python-3.6.8
else
 module load Stages/2020
 ml GCC/9.3.0 
 ml ParaStationMPI/5.4.7-1 
 ml CDO/1.9.8
 ml mpi4py/3.0.3-Python-3.8.5
 echo "I am here"
 ml SciPy-Stack/2020-Python-3.8.5
 ml scikit/2020-Python-3.8.5
fi
#ml SciPy-Stack/2019a-Python-3.6.8
#ml scikit/2019a-Python-3.6.8
#ml netcdf4-python/1.5.0.1-Python-3.6.8

# clean up if triggered via script argument
if [[ $1 == purge ]]; then
  echo "Purge all modules after loading them..."
  module --force purge
fi  
