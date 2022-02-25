#!/bin/bash -x

#your virtual enviornment name
VIRT_ENV_NAME=venv_test

echo "Activating virtual environment..."
source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate

# the source directory contains the tfrecords from data preprocessing step 2
source_dir=/path/to/tfrecords
destination_dir=/path/to/output/directory

#select models
model=savp
mkdir ${destination_dir}
cp ../hparams/era5/${model}/model_hparams_template.json ${destination_dir}/model_hparams.json
cp ../data_split/era5/datasplit.json ${destination_dir}/data_split.json

#copy the configuration to destination_dir
vim ${destination_dir}/data_split.json
vim ${destination_dir}/model_hparams.json

datasplit_dict=${destination_dir}/data_split.json
model_hparams=${destination_dir}/model_hparams.json

python3 ../main_scripts/main_train_models.py --input_dir ${source_dir} --datasplit_dict ${datasplit_dict} \
     --dataset era5 --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/



