#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=preprocess-out.%j
#SBATCH --error=preprocess-err.%j
#SBATCH --time=20:20:00
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

module --force purge 
module /usr/local/software/jureca/OtherStages
module load Stages/2019a
module load GCCcore/.8.3.0
module load mpi4py/3.0.1-Python-3.6.8
module load h5py/2.9.0-serial-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8

srun bash data/download_and_preprocess_dataset_era5.sh --data era5 --input_dir /p/scratch/deepacf/bing/processData_size_64_64_3/splits --output_dir data/era5_size_64_64_3/ours_savp
