#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --output=log_out.%j
#SBATCH --error=log_err.%j
#SBATCH --time=00:10:00
#SBATCH --partition=batch

ml Stages/2022
ml GCCcore/.11.2.0
ml GCC/11.2.0
ml ParaStationMPI/5.5.0-1

ml Python/3.9.6
ml SciPy-bundle/2021.10
ml xarray/0.20.1
ml netcdf4-python/1.5.7
ml dask/2021.9.1

source_dir=/p/scratch/deepacf/inbound_data/weatherbench
dest_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/weatherbench_test

data_extraction_dir=/p/project/deepacf/deeprain/grasse/ambs/video_prediction_tools/data_preprocess

cd ${data_extraction_dir}

# Name of virtual environment 
venv_dir=".venv"
python -m venv --system-site-packages ${venv_dir}
. ${venv_dir}/bin/activate
#pip3 install --no-cache-dir pytz
#pip3 install --no-cache-dir python-dateutil
export PYTHONPATH=${data_extraction_dir}:$PYTHONPATH
export PYTHONPATH="${data_extraction_dir}/..":$PYTHONPATH

python3 ../main_scripts/main_data_extraction.py ${source_dir} ${dest_dir} -1

rm -r ${venv_dir}