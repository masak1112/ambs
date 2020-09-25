#!/bin/bash -x

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/home/${USER}/preprocessedData/
destination_dir=/home/${USER}/models/

# for choosing the model
model=mcnet
model_hparams=../hparams/era5/model_hparams.json

# execute respective Python-script
python ../scripts/train_dummy.py --input_dir  ${source_dir}/tfrecords/ --dataset era5  --model ${model} --model_hparams_dict ${model_hparams} --output_dir ${destination_dir}/${model}/
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
