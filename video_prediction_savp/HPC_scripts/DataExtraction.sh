#!/bin/bash -x


#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=DataExtraction-out.%j
#SBATCH --error=DataExtraction-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de


jutil env activate -p deepacf
module purge
module use $OTHERSTAGES
module load Stages/2019a
# on HDFML: h5py requires Gcc/8.3.0 and not Intel/2019.3.199-GCC-8.3.0 
#module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load GCC/8.3.0 ParaStationMPI/5.2.2-1
module load h5py/2.9.0-Python-3.6.8
module load mpi4py/3.0.1-Python-3.6.8
module load netcdf4-python/1.5.0.1-Python-3.6.8

srun python ../../workflow_parallel_frame_prediction/DataExtraction/mpi_stager_v2.py --source_dir /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/2017/ --destination_dir /p/scratch/deepacf/${USER}/extractedData/2017


# 2tier pystager 
#srun python ../../workflow_parallel_frame_prediction/DataExtraction/main_single_master.py --source_dir /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/2017/ --destination_dir /p/scratch/deepacf/${USER}/extractedData/2017

