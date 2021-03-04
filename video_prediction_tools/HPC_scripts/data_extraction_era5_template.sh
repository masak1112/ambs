#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=data_extraction_era5-out.%j
#SBATCH --error=data_extraction_era5-err.%j
#SBATCH --time=04:20:00
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
source ../env_setup/modules_data_extraction.sh
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
source_dir="/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib"
dest_dir="/p/project/deepacf/deeprain/video_prediction_shared_folder/extractedData/"
varslist_path="/p/home/jusers/gong1/juwels/ambs/video_prediction_tools/data_split/data_extraction_era5.json"

# Run data extraction

declare -a years=(
                 "2015"
                 "2016"
                 "2017"
                  )

# Run data extraction
for year in "${years[@]}";     do
        echo "Year $year"
        srun python ../main_scripts/main_data_extraction.py  --source_dir ${source_dir} --target_dir ${dest_dir} --year ${year} --varslist_path ${varslist_path}

done2tier pystager 
#srun python ../../workflow_parallel_frame_prediction/DataExtraction/main_single_master.py --source_dir /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/${year}/ --destination_dir ${SAVE_DIR}/extractedData/${year}
