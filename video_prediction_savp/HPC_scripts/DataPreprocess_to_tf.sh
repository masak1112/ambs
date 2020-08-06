#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_to_tf-out.%j
#SBATCH --error=DataPreprocess_to_tf-err.%j
#SBATCH --time=01:20:00
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de


# Name of virtual environment 
VIRT_ENV_NAME="vp"

# Loading mouldes
source ../env_setup/modules_train.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../${VIRT_ENV_NAME}/bin/activate
   else 
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/scarlet_era5-Y2017_testM01to12-160x128-2970N1500W-T2_MSL_gph500
destination_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/bing_era5-Y2017_testM01to12-160x128-2970N1500W-T2_MSL_gph500

# run Preprocessing (step 2 where Tf-records are generated)
srun python ../video_prediction/datasets/era5_dataset_v2.py ${source_dir}/hickle ${destination_dir}/tfrecords -vars T2 MSL gph500 -height 128 -width 160 -seq_length 20 
