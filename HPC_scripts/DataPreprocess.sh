#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess-out.%j
#SBATCH --error=DataPreprocess-err.%j
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

module --force purge 
module use $OTHERSTAGES
module load Stages/2019a
module load GCCcore/.8.3.0
module load mpi4py/3.0.1-Python-3.6.8
module load h5py/2.9.0-serial-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load cuDNN/7.5.1.10-CUDA-10.1.105


srun python video_prediction/datasets/era5_dataset_v2.py /p/scratch/deepacf/bing/processData_size_64_64_3_2016/splits/ ./data/era5_size_64_64_3_norm_2016

#srun python scripts/generate_transfer_learning.py --input_dir data/era5_size_64_64_1_v2 --dataset_hparams sequence_length=20  --checkpoint pretrained_models/kth/ours_savp  --mode test --results_dir results_test_samples/era5_size_64_64_1_v2_pretrained --dataset era5
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
