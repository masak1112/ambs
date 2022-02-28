#!/bin/bash -x

#User's input : your virtual enviornment name
VIRT_ENV_NAME=venv_test
# !!! ADAPAT DEPENDING ON USAGE OF CONTAINER !!!
# For container usage, comment in the follwoing lines
# Name of container image (must be available in working directory)
CONTAINER_IMG="${WORK_DIR}/tensorflow_21.09-tf1-py3.sif"
WRAPPER="${BASE_DIR}/env_setup/wrapper_container.sh"

# sanity checks
if [[ ! -f ${CONTAINER_IMG} ]]; then
  echo "ERROR: Cannot find required TF1.15 container image '${CONTAINER_IMG}'."
  exit 1
fi

if [[ ! -f ${WRAPPER} ]]; then
  echo "ERROR: Cannot find wrapper-script '${WRAPPER}' for TF1.15 container image."
  exit 1
fi

# declare directory-variables which will be modified by generate_runscript.py
# Note: source_dir is only needed for retrieving the base-directory
checkpoint_dir=/my/trained/model/dir
results_dir=/my/results/dir
lquick=""

# run postprocessing/generation of model results including evaluation metrics
export CUDA_VISIBLE_DEVICES=0

# For running on small datasets (e.g. the dry run), parse -test to the Python-script
singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} \
python3 ../main_scripts/main_visualize_postprocess.py --checkpoint  ${checkpoint_dir} --mode test  \
                                                      --results_dir ${results_dir} --batch_size 4 \
                                                      --num_stochastic_samples 1 \
  					                                          --lquick_evaluation --climatology_file ${climate_file}


# WITHOUT container usage, comment in the follwoing lines (and uncomment the lines above)
# Activate virtual environment if needed (and possible)
#if [ -z ${VIRTUAL_ENV} ]; then
#   if [[ -f ../virtual_envs/${VIRT_ENV_NAME}/bin/activate ]]; then
#      echo "Activating virtual environment..."
#      source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate
#   else
#      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
#      exit 1
#   fi
#fi
#
# Run postprocessing
# For running on small datasets (e.g. the dry run), parse -test to the Python-script
# python3 ../main_scripts/main_visualize_postprocess.py --checkpoint  ${checkpoint_dir} --mode test  \
#                                                      --results_dir ${results_dir} --batch_size 4 \
#                                                      --num_stochastic_samples 1 \
#							                                        --lquick_evaluation --climatology_file ${climate_file}

