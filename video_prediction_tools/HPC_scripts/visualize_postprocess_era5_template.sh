#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=postprocess_era5-out.%j
#SBATCH --error=postprocess_era5-err.%j
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de
##jutil env activate -p cjjsc42

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment 
VIRT_ENV_NAME="my_venv"

# Loading modules
source ../env_setup/modules_postprocess.sh
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
# Note: source_dir is only needed for retrieving the base-directory
source_dir=/my/source/dir/
checkpoint_dir=/my/trained/model/dir
results_dir=/my/results/dir

# name of model
model=convLSTM

# run postprocessing/generation of model results including evaluation metrics
srun python -u ../main_scripts/main_visualize_postprocess.py --checkpoint  ${checkpoint_dir} --mode test  \
                                                             --results_dir ${results_dir} --batch_size 4 \
                                                             --num_stochastic_samples 1  \
                                                               > postprocess_era5-out_all.${SLURM_JOB_ID}
