#!/bin/bash -x


python -u ../scripts/generate_transfer_learning_finetune.py \
--input_dir /home/${USER}/preprocessedData/era5-Y2015toY2017M01to12-128x160-74d00N71d00E-T_MSL_gph500/tfrecords  \
--dataset_hparams sequence_length=20 --checkpoint  /home/${USER}/models/era5-Y2015toY2017M01to12-128x160-74d00N71d00E-T_MSL_gph500/ours_savp \
--mode test --results_dir /home/${USER}/results/era5-Y2015toY2017M01to12-128x160-74d00N71d00E-T_MSL_gph500 \
--batch_size 2 --dataset era5   > generate_era5-out.out

#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
