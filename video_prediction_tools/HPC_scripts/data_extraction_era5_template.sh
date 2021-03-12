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
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

jutil env activate -p deepacf

# Name of virtual environment 
VIRT_ENV_NAME="my_venv"

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

# Declare path-variables (dest_dir will be set and configured automatically via generate_runscript.py)
source_dir=/my/path/to/era5
destination_dir=/my/path/to/extracted/data
varmap_file=/my/path/to/varmapping/file

years=( "2015" )

# Run data extraction
for year in "${years[@]}"; do
  echo "Perform ERA5-data extraction for year ${year}"
  srun python ../main_scripts/main_data_extraction.py  --source_dir ${source_dir} --target_dir ${destination_dir} \
                                                       --year ${year} --varslist_path ${varslist_path}
done
