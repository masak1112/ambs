#!/usr/bin/env bash
set -e

MODEL=savp
#train_mode: end_to_end, pre_trained, fine_tune
TRAIN_MODE=end_to_end
EXP_NAME=era5_size_64_64_3_3t_norm

DATA_ETL_DIR=/p/scratch/deepacf/{USER}/
DATA_EXTRA=${DATA_ETL_DIR}/extractedData/${EXP_NAME}
DATA_PREPROCESS=${DATA_ETL_DIR}/preprocessedData/${EXP_NAME}
DATA_PREPROCESS_TF=./data/${EXP_NAME}
TRAIN_OUTPUT=./logs/${EXP_NAME}/${TRAIN_MODE}
RESULTS_OUTPUT=./results_test_samples/${EXP_NAME}/${TRAIN_MODE}/

if [ $MODEL==savp ]
then
    method_dir=ours_savp
elif [ $MODEL==gan ]
then
    method_dir=ours_gan
elif [ $MODEL==vae ]
then
    method_dir=ours_vae
else
    echo "model does not exist" 2>&1
    exit 1
fi


##############Datat Preprocessing################
#To hkl data
python ../workflow_video_prediction/DataPreprocess/benchmark/mpi_stager_v2_process_netCDF.py \
--input_dir $DATA_EXTRA --destination_dir $DATA_PREPROCESS
#Change the .hkl data to .tfrecords files
python ./video_prediction/datasets/era5_dataset_v2.py  --source_dir $DATA_PREPROCESS/splits \
--destination_dir $DATA_PREPROCESS_TF

#########Train##########################
python ./scripts/train_v2.py --input_dir $DATA_PREPROCESS_TF --dataset era5  \
--model $MODEL --model_hparams_dict hparams/kth/${method_dir}/model_hparams.json \
--output_dir ${TRAIN_OUTPUT}


#########Generate results#################
python ./scripts/generate_transfer_learning_finetune.py --input_dir ${DATA_PREPROCESS_TF} \
--dataset_hparams sequence_length=20 --checkpoint $TRAIN_OUTPUT\
--mode test --results_dir ${RESULTS_OUTPUT} \
--batch_size 4 --dataset era5

