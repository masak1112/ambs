#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=13
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_era5_step2-out.%j
#SBATCH --error=DataPreprocess_era5_step2-err.%j
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment 
VIRT_ENV_NAME="my_venv"

# Loading mouldes
source ../env_setup/modules_train.sh
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

# declare directory-variables which will be modified by config_runscript.py
source_dir=/my/path/to/pkl/files/
destination_dir=/my/path/to/tfrecords/files

sequence_length=20
sequences_per_file=10
# run Preprocessing (step 2 where Tf-records are generated)
srun python ../main_scripts/main_preprocess_data_step2.py -source_dir ${source_dir} -dest_dir ${destination_dir} \
            -sequence_length ${sequence_length} -sequences_per_file ${sequences_per_file}
