#!/bin/bash -x

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/home/${USER}/preprocessedData/
destination_dir=/home/${USER}/models/

model=savp

python ../scripts/train_v2.py --input_dir  ${source_dir}/tfrecords/ --dataset era5  --model ${model} --model_hparams_dict ../hparams/kth/ours_savp/model_hparams.json --output_dir ${destination_dir}/${model}/
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
