#!/bin/bash -x



python ../scripts/train_v2.py --input_dir  /home/${USER}/preprocessedData/era5-Y2015toY2017M01to12-128x160-74d00N71d00E-T_MSL_gph500/tfrecords --dataset era5  --model savp --model_hparams_dict ../hparams/kth/ours_savp/model_hparams.json --output_dir /home/${USER}/models/era5-Y2015toY2017M01to12-128x160-74d00N71d00E-T_MSL_gph500/ours_savp 
#srun  python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
