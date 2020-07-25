#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_era5-out.%j
#SBATCH --error=train_era5-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

# Name of virtual environment 
VIRT_ENV_NAME="virt_env_hdfml"

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
source_dir=/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/era5-Y2009Y2012to2013Y2016Y2019M01to12-160x128-2970N1500W-T2_MSL_gph500/era5-Y2010Y2022M01to12-160x128-2970N1500W-T2_MSL_gph500/era5-Y2010Y2022M01to12-160x128-2970N1500W-T2_MSL_gph500/era5-Y2015to2017M01to12-160x128-2970N1500W-T2_MSL_gph500/
destination_dir=/p/scratch/deepacf/video_prediction_shared_folder/models/era5-Y2009Y2012to2013Y2016Y2019M01to12-160x128-2970N1500W-T2_MSL_gph500/era5-Y2010Y2022M01to12-160x128-2970N1500W-T2_MSL_gph500/era5-Y2010Y2022M01to12-160x128-2970N1500W-T2_MSL_gph500/era5-Y2015to2017M01to12-160x128-2970N1500W-T2_MSL_gph500/

# for choosing the model
model=mcnet
model_hparams=../hparams/era5/model_hparams.json

# rund training
srun python ../scripts/train_dummy.py --input_dir  ${source_dir}/tfrecords/ --dataset era5  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/${model}/
 
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
