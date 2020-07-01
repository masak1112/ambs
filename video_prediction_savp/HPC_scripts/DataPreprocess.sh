#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess-out.%j
#SBATCH --error=DataPreprocess-err.%j
#SBATCH --time=02:20:00
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

if [ -z ${VIRTUAL_ENV} ]; then
  echo "Please activate a virtual environment..."
  exit 1
fi

source ../env_setup/modules_preprocess.sh

source_dir=/p/scratch/deepacf/video_prediction_shared_folder/extractedData
destination_dir=/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/era5-Y2015to2017M01to12
script_dir=`pwd`

declare -a years=("2015" 
                 "2016" 
                 "2017"
                  )

# ececute Python-scripts
for year in "${years[@]}"; 
    do 
        echo "Year $year"
	echo "source_dir ${source_dir}/${year}"
	srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
        --source_dir ${source_dir} -scr_dir ${script_dir} \
        --destination_dir ${destination_dir} --years ${year} --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710     
    done


srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500    
