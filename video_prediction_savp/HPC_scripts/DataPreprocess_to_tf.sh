#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_to_tf-out.%j
#SBATCH --error=DataPreprocess_to_tf-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de


if [ -z ${VIRTUAL_ENV} ]; then
  echo "Please activate a virtual environment..."
  exit 1
fi

source ../env_setup/modules_train.sh

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/
destination_dir=/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/

# execute Python-script
srun python ../video_prediction/datasets/era5_dataset_v2.py ${source_dir}/hickle/splits ${destination_dir}/tfrecords -vars T2 MSL gph500 -height 128 -width 160 -seq_length 20 
