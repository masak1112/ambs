#!/usr/bin/env bash
set -e
#
#MODEL=savp
##train_mode: end_to_end, pre_trained
#TRAIN_MODE=end_to_end
#EXP_NAME=era5_size_64_64_3_3t_norm

MODEL=$1
TRAIN_MODE=$2
EXP_NAME=$3

DATA_ETL_DIR=/p/scratch/deepacf/${USER}/
DATA_EXTRA_DIR=${DATA_ETL_DIR}/extractedData/${EXP_NAME}
DATA_PREPROCESS_DIR=${DATA_ETL_DIR}/preprocessedData/${EXP_NAME}
DATA_PREPROCESS_TF_DIR=./data/${EXP_NAME}
RESULTS_OUTPUT_DIR=./results_test_samples/${EXP_NAME}/${TRAIN_MODE}/

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

if [ ${TRAIN_MODE}==pre_trained ]; then
    TRAIN_OUTPUT_DIR=./pretrained_models/kth/${method_dir}
else
    TRAIN_OUTPUT_DIR=./logs/${EXP_NAME}/${TRAIN_MODE}
fi

##############Datat Preprocessing################
#To hkl data
if [ -d ${DATA_PREPROCESS_DIR} ]; then
    echo "The Preprocessed Data (.hkl ) exist"
else
    python ../workflow_video_prediction/DataPreprocess/benchmark/mpi_stager_v2_process_netCDF.py \
    --input_dir ${DATA_EXTRA_DIR} --destination_dir ${DATA_PREPROCESS_DIR}
fi

#Change the .hkl data to .tfrecords files
if [ -d ${DATA_PREPROCESS_TF_DIR} ]; then
    echo "The Preprocessed Data (tf.records) exist"
else
    python ./video_prediction/datasets/era5_dataset_v2.py  --source_dir ${DATA_PREPROCESS_DIR}/splits \
    --destination_dir ${DATA_PREPROCESS_TF_DIR}
fi

#########Train##########################
if [ ${TRAIN_MODE}==pre_trained ]; then
    echo "Using kth trained model "
else
    python ./scripts/train_v2.py --input_dir $DATA_PREPROCESS_TF_DIR --dataset era5  \
    --model ${MODEL} --model_hparams_dict hparams/kth/${method_dir}/model_hparams.json \
    --output_dir ${TRAIN_OUTPUT_DIR}
fi

#########Generate results#################
python ./scripts/generate_transfer_learning_finetune.py --input_dir ${DATA_PREPROCESS_TF_DIR} \
--dataset_hparams sequence_length=20 --checkpoint $TRAIN_OUTPUT_DIR\
--mode test --results_dir ${RESULTS_OUTPUT_DIR} \
--batch_size 4 --dataset era5