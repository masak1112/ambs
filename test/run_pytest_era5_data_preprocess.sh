

# Name of virtual environment
VIRT_ENV_NAME="venv_hdfml"


CONTAINER_IMG="../video_prediction_tools/HPC_scripts/tensorflow_21.09-tf1-py3.sif"
WRAPPER="./wrapper_container.sh"

# sanity checks
if [[ ! -f ${CONTAINER_IMG} ]]; then
  echo "ERROR: Cannot find required TF1.15 container image '${CONTAINER_IMG}'."
  exit 1
fi

if [[ ! -f ${WRAPPER} ]]; then
  echo "ERROR: Cannot find wrapper-script '${WRAPPER}' for TF1.15 container image."
  exit 1
fi

#source ../video_prediction_tools/env_setup/modules_preprocess+extract.sh
singularity exec --nv "${CONTAINER_IMG}" "${WRAPPER}" ${VIRT_ENV_NAME} python3 -m pytest test_era5_data.py

