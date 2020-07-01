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
#SBATCH --mail-user=m.langguth@fz-juelich.de
##jutil env activate -p cjjsc42

if [ -z ${VIRTUAL_ENV} ]; then
  echo "Please activate a virtual environment..."
  exit 1
fi

source ../env_setup/modules_train.sh

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/
destination_dir=/p/scratch/deepacf/video_prediction_shared_folder/models/

# for choosing the model
model=mcnet
model_hparams=../hparams/era5/model_hparams.json

# execute Python-script
srun python ../scripts/train_dummy.py --input_dir  ${source_dir}/tfrecords/ --dataset era5  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/${model}/

 
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
