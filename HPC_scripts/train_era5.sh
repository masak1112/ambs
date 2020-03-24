#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=train_era5-out.%j
#SBATCH --error=train_era5-err.%j
#SBATCH --time=23:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

module --force purge 
module use use $OTHERSTAGES
module load Stages/2019a
module load GCCcore/.8.3.0
module load mpi4py/3.0.1-Python-3.6.8
module load h5py/2.9.0-serial-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load cuDNN/7.5.1.10-CUDA-10.1.105

#srun  python scripts/train.py --input_dir data/kth --dataset kth  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/kth/ours_savp

srun python ../scripts/train_v2.py --input_dir ../data/era5_size_64_64_3_3t_norm --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir ../logs/era5_size_64_64_3_3t_norm/ours_savp
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
