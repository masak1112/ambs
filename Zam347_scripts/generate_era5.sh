#!/bin/bash -x

# declare directory-variables which will be modified appropriately during Preprocessing (invoked by mpi_split_data_multi_years.py)
source_dir=/home/${USER}/preprocessedData/
checkpoint_dir=/home/${USER}/models/
results_dir=/home/${USER}/results/

# for choosing the model
model=mcnet

# execute respective Python-script
python -u ../scripts/generate_transfer_learning_finetune.py \
--input_dir ${source_dir}/tfrecords  \
--dataset_hparams sequence_length=20 --checkpoint  ${checkpoint_dir}/${model} \
--mode test --results_dir ${results_dir} \
--batch_size 2 --dataset era5   > generate_era5-out.out

#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
