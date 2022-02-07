#!/usr/bin/env bash

# __author__ = Michael Langguth
# __date__  = '2022_02_01'

# This script loads the required modules for AMBS on JSC's HPY_systems (HDF-ML, Juwels Cluster and Juwels Booster).
# Further Python-packages may be installed in the virtual environment created by create_env.sh
# (see also requirements.txt).

HOST_NAME=$(hostname)

echo "Start loading modules on ${HOST_NAME} required for preprocessing..."
echo "modules_preprocess+extract.sh is used for: "
echo "* data_extraction_era5.sh"
echo "* preprocess_data_era5_step1.sh"
echo "* generate_runscript.py"

module purge
module use "$OTHERSTAGES"
ml Stages/2020
ml GCC/10.3.0
ml GCCcore/.10.3.0
ml ParaStationMPI/5.4.10-1
ml mpi4py/3.0.3-Python-3.8.5
ml mpi4py/3.0.1-Python-3.6.8
ml h5py/2.10.0-Python-3.8.5
ml netcdf4-python/1.5.4-Python-3.8.5
ml SciPy-Stack/2021-Python-3.8.5
ml scikit/2021-Python-3.8.5
ml CDO/2.0.0rc3

# clean up if triggered via script argument
if [[ $1 == purge ]]; then
  echo "Purge all modules after loading them..."
  module --force purge
fi  
