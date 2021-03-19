#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/home/${USER}/preprocessedData/
destination_dir=/home/${USER}/models/

# valid identifiers for model-argument are: convLSTM, savp, mcnet and vae
model=mcnet
model_hparams=../hparams/era5/model_hparams.json
destination_dir=${destination_dir}/${model}/"$(date +"%Y%m%dT%H%M")_"$USER"/"

# run training
python ../scripts/train_dummy.py --input_dir  ${source_dir}/tfrecords/ --dataset era5  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}

