#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# clean-up modules to avoid conflicts between host and container settings
module purge

# declare directory-variables which will be modified by generate_runscript.py
# Note: source_dir is only needed for retrieving the base-directory
source_dir=/my/source/dir/
checkpoint_dir=/my/trained/model/dir
results_dir=/my/results/dir
lquick=""

# run postprocessing/generation of model results including evaluation metrics
export CUDA_VISIBLE_DEVICES=0
## One node, single GPU
srun --mpi=pspmix --cpu-bind=none \
     singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} \
     python3 ../main_scripts/main_visualize_postprocess.py --checkpoint  ${checkpoint_dir} --mode test  \
                                                           --results_dir ${results_dir} --batch_size 4 \
                                                           --num_stochastic_samples 1 ${lquick} \
                                                           > postprocess_era5-out_all."${SLURM_JOB_ID}"

