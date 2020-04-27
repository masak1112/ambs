#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=data_preprocess_tf-out.%j
#SBATCH --error=data_preprocess_tf-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de

module --force purge
module $OTHERSTAGES
module load Stages/2019a
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load h5py/2.9.0-Python-3.6.8
module load mpi4py/3.0.1-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8

srun python ../video_prediction/datasets/era5_dataset_v2.py  /p/scratch/deepacf/video_prediction_shared_folder/processData/splits/  ../data/era5_64_64_3_3t_norm

