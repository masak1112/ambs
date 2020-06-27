#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=DataExtraction-out.%j
#SBATCH --error=DataExtraction-err.%j
#SBATCH --time=05:00:00
##SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.langguth@fz-juelich.de


jutil env activate -p deepacf

if [ -z ${VIRTUAL_ENV} ]; then
  echo "Please activate a virtual environment..."
  exit 1
fi

source ../env_setup/module.sh

srun python ../../workflow_parallel_frame_prediction/DataExtraction/mpi_stager_v2.py --source_dir /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/2017/ --destination_dir /p/scratch/deepacf/${USER}/extractedData/2017


# 2tier pystager 
#srun python ../../workflow_parallel_frame_prediction/DataExtraction/main_single_master.py --source_dir /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/2017/ --destination_dir /p/scratch/deepacf/${USER}/extractedData/2017

