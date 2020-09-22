#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=data_extraction_era5-out.%j
#SBATCH --error=data_extraction_era5-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

jutil env activate -p deepacf

# Name of virtual environment 
VIRT_ENV_NAME="virt_env_hdfml"

# Loading mouldes
source ../env_setup/modules_preprocess.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../${VIRT_ENV_NAME}/bin/activate
   else 
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

# Declare path-variables
source_dir="/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/"
dest_dir="/p/scratch/deepacf/video_prediction_shared_folder/extractedData/"

year="2010"

# Run data extraction
srun python ../main_scripts/main_data_extraction.py  --source_dir ${source_dir}/${year}/ --destination_dir ${dest_dir}/${year}/



# 2tier pystager 
#srun python ../../workflow_parallel_frame_prediction/DataExtraction/main_single_master.py --source_dir /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/${year}/ --destination_dir ${SAVE_DIR}/extractedData/${year}
