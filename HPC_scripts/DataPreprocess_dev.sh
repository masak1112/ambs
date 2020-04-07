#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess-out.%j
#SBATCH --error=DataPreprocess-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

module --force purge
module use  $OTHERSTAGES
module load Stages/2019a
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load h5py/2.9.0-Python-3.6.8
module load mpi4py/3.0.1-Python-3.6.8

srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
 --source_dir /p/scratch/deepacf/video_prediction_shared_folder/extractedData/2016 \
 --destination_dir /p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/Y2016toY2017M01to12-128x160-74d00N71d0E-T_MSL_gph500/2016 \
 --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710



#srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
# --source_dir /p/scratch/deepacf/video_prediction_shared_folder/extractedData/2017 \
# --destination_dir /p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/Y2016toY2017M01to12-128x160-74d00N71d0E-T_MSL_gph500/2017 \
# --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710
