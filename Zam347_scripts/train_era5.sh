#!/bin/bash -x



python ../scripts/train_dummy.py --input_dir  /home/${USER}/preprocessedData/era5-Y2017M01to02-128x160-74d00N71d00E-T_MSL_gph500/tfrecords --dataset era5  --model mcnet  --model_hparams_dict ../hparams/era5/model_hparams.json --output_dir /home/${USER}/models/era5-Y2017M01to02-128x160-74d00N71d00E-T_MSL_gph500/mcnet
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
