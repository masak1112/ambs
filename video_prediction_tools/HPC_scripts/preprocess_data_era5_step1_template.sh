#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=<your_project>
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_era5_step1-out.%j
#SBATCH --error=DataPreprocess_era5_step1-err.%j
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
if [ -z ${VIRTUAL_ENV} ]; then
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
years=( "2015" )
variables=( "t2" "t2" "t2" )
sw_corner=( -999.9 -999.9)
nyx=( -999 -999 )

# set some paths
# note, that destination_dir is adjusted during runtime based on the data
source_dir=/my/path/to/extracted/data/
destination_dir=/my/path/to/pickle/files

# execute Python-scripts
for year in "${years[@]}"; do
  echo "start preprocessing data for year ${year}"
	srun python ../main_scripts/main_preprocess_data_step1.py \
        --source_dir ${source_dir} --destination_dir ${destination_dir} --years "${year}" \
       	--vars "${variables[0]}" "${variables[1]}" "${variables[2]}" \
       	--sw_corner "${sw_corner[0]}" "${sw_corner[1]}" --nyx "${nyx[0]}" "${nyx[1]}"
done


#srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500    
