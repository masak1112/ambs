#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=3
##SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_to_tf-out.%j
#SBATCH --error=DataPreprocess_to_tf-err.%j
#SBATCH --time=00:50:00
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

module purge
module use $OTHERSTAGES
module load Stages/2019a
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load h5py/2.9.0-Python-3.6.8
module load mpi4py/3.0.1-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8


source_dir=/p/scratch/deepacf/video_prediction_shared_folder/preproceesedData
destination_dir=/p/scratch/deepacf/video_prediction_shared_folder/preproceesedData


srun python ../video_prediction/datasets/era5_dataset_v2.py ${source_dir}/hickle/splits ${destination_dir}/tfrecords -vars T2 MSL gph500 -height 128 -width 160 -seq_length 20 
