#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=process_netcdf-out.%j
#SBATCH --error=process_netcdf-err.%j
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

srun python mpi_stager_v2_process_netCDF.py --source_dir /p/scratch/deepacf/video_prediction_shared_folder/extractedData/\
--destination_dir /p/scratch/deepacf/bing/processData_size_64_64_3_3t_norm_test2
