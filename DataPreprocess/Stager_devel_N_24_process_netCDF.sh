#!/bin/bash -x
#SBATCH --account=jjsc42
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
##jutil env activate -p cjjsc42

module --force purge 
module /usr/local/software/jureca/OtherStages
module load Stages/2019a
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1-mt
module load mpi4py/3.0.1-Python-3.6.8

srun python mpi_stager_v2_process_netCDF.py
