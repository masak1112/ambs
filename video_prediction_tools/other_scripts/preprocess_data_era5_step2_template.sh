#!/bin/bash -x

#User's input : your virtual enviornment name
VIRT_ENV_NAME=venv_test

echo "Activating virtual environment..."
source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate

sequence_length=20
sequences_per_file=10

source_dir=/home/b.gong/preprocessed_data-24x24-990N2010E-var167/pickle
base_dir="$(dirname "$source_dir")"
destination_dir=${base_dir}/tfrecords
n_nodes=2

mpirun -n ${n_nodes}  python3 ../main_scripts/main_preprocess_data_step2.py -source_dir ${source_dir} -dest_dir ${destination_dir} \
     -sequence_length ${sequence_length} -sequences_per_file ${sequences_per_file}





