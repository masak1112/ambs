#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=generate_era5-out.%j
#SBATCH --error=generate_era5-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42



module load GCC/8.3.0
module load ParaStationMPI/5.2.2-1
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load netcdf4-python/1.5.0.1-Python-3.6.8
module load h5py/2.9.0-Python-3.6.8
source mandarine/bin/activate

python scripts/generate_transfer_learning_finetune.py --input_dir data/era5_size_64_64_3_norm_dup --dataset_hparams sequence_length=20 --checkpoint logs/era5_64_64_3_norm_2016/ours_savp --mode test --results_dir results_test_samples/era5_size_64_64_3_norm_2016  --batch_size 4 --dataset era5 
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
