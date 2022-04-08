#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=<your_project>
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataExtraction_era5_step1-out.%j
#SBATCH --error=DataExtraction_era5_step1-err.%j
#SBATCH --time=04:20:00
#SBATCH --gres=gpu:0
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment 
VIRT_ENV_NAME="my_venv"

# Activate virtual environment if needed (and possible)
if [ -z "${VIRTUAL_ENV}" ]; then
   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
   else 
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi
# Loading modules
source ../env_setup/modules_preprocess+extract.sh


# select years and variables for dataset and define target domain 
years=( 2017 )
months=( "all" )
var_dict='{"2t": {"sf": ""}, "tcc": {"sf": ""}, "t": {"ml": "p85000."}}'
sw_corner=(38.4 0.0)
nyx=(56 92)

# set some paths
# note, that destination_dir is adjusted during runtime based on the data
source_dir=/my/path/to/era5/data
destination_dir=/my/path/to/extracted/data

# execute Python-script
srun python ../main_scripts/main_era5_data_extraction.py -src_dir "${source_dir}" \
     -dest_dir "${destination_dir}" -y "${years[@]}" -m "${months[@]}" \
     -swc "${sw_corner[@]}" -nyx "${nyx[@]}" -v "${var_dict}"

