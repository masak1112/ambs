#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_to_tf-out.%j
#SBATCH --error=DataPreprocess_to_tf-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

module --force purge
module use $OTHERSTAGES
module load Stages/2019a
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load h5py/2.9.0-Python-3.6.8
module load mpi4py/3.0.1-Python-3.6.8


srun python ../video_prediction/datasets/era5_dataset_v2.py /p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/2017M01to12-64_64-50.00N11.50E-T_T_T/hickle/splits/ /p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/2017M01to12-64_64-50.00N11.50E-T_T_T/tfrecords/ 
