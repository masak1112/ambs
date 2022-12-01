#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --output=log_out.%j
#SBATCH --error=log_err.%j
#SBATCH --time=00:10:00
#SBATCH --partition=batch

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

ml Stages/2022
ml GCCcore/.11.2.0
ml GCC/11.2.0
ml ParaStationMPI/5.5.0-1

ml Python/3.9.6
ml SciPy-bundle/2021.10
ml xarray/0.20.1
ml netcdf4-python/1.5.7
ml dask/2021.9.1

# Name of virtual environment 
VIRT_ENV_NAME="my_venv"

# Activate virtual environment if needed (and possible)
"""
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
   else 
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi
# Loading modules
source ../env_setup/modules_preprocess+extract.sh
"""

source_dir=/p/scratch/deepacf/inbound_data/weatherbench
destination_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/weatherbench_test/extracted
data_extraction_dir=/p/project/deepacf/deeprain/grasse/ambs/video_prediction_tools/data_preprocess
variables='[{"name":"temperature","lvl":[850],"interpolation":"p"},{"name":"geopotential","lvl":[500],"interpolation":"p"}]'
years=("2013" "2014" "2015" "2016" "2017")

cd ${data_extraction_dir}

# Name of virtual environment 
venv_dir=".venv"
python -m venv --system-site-packages ${venv_dir}
. ${venv_dir}/bin/activate
#pip3 install --no-cache-dir pytz
#pip3 install --no-cache-dir python-dateutil
export PYTHONPATH=${data_extraction_dir}:$PYTHONPATH
export PYTHONPATH="${data_extraction_dir}/..":$PYTHONPATH

python3 ../main_scripts/main_data_extraction.py ${source_dir} ${dest_dir} ${years[@]} ${variables}

rm -r ${venv_dir}
