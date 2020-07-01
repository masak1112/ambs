#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_era5-out.%j
#SBATCH --error=train_era5-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:2
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

module  purge 
module use $OTHERSTAGES
module load Stages/2019a
module load GCCcore/.8.3.0
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load h5py/2.9.0-serial-Python-3.6.8
module load mpi4py/3.0.1-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load cuDNN/7.5.1.10-CUDA-10.1.105


# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/
destination_dir=/p/scratch/deepacf/video_prediction_shared_folder/models/

#define model type, hyperparams setting up
#source hyperparam_dir.sh 

srun python ../scripts/train_dummy.py --input_dir  ${source_dir}/tfrecords/ --dataset era5  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/${model}/${hyperdir}
 
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
