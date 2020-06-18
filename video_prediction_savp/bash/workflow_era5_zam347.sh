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
RETRAIN=1 #if we continue training the model or using the existing end-to-end model, 1 means continue training, and 1 means use the existing one
DATA_ETL_DIR=/home/${USER}/
DATA_ETL_DIR=/p/scratch/deepacf/${USER}/
DATA_EXTRA_DIR=${DATA_ETL_DIR}/extractedData/${EXP_NAME}
DATA_PREPROCESS_DIR=${DATA_ETL_DIR}/preprocessedData/${EXP_NAME}
DATA_PREPROCESS_TF_DIR=./data/${EXP_NAME}
RESULTS_OUTPUT_DIR=./results_test_samples/${EXP_NAME}/${TRAIN_MODE}/

if [ $MODEL==savp ]; then
    method_dir=ours_savp
elif [ $MODEL==gan ]; then
    method_dir=ours_gan
elif [ $MODEL==vae ]; then
    method_dir=ours_vae
else
    echo "model does not exist" 2>&1
    exit 1
fi

if [ "$TRAIN_MODE" == pre_trained ]; then
    TRAIN_OUTPUT_DIR=./pretrained_models/kth/${method_dir}
else
    TRAIN_OUTPUT_DIR=./logs/${EXP_NAME}/${TRAIN_MODE}
fi

CHECKPOINT_DIR=${TRAIN_OUTPUT_DIR}/${method_dir}

echo "===========================WORKFLOW SETUP===================="
echo "Model ${MODEL}"
echo "TRAIN MODE ${TRAIN_MODE}"
echo "Method_dir ${method_dir}"
echo "DATA_ETL_DIR ${DATA_ETL_DIR}"
echo "DATA_EXTRA_DIR ${DATA_EXTRA_DIR}"
echo "DATA_PREPROCESS_DIR ${DATA_PREPROCESS_DIR}"
echo "DATA_PREPROCESS_TF_DIR ${DATA_PREPROCESS_TF_DIR}"
echo "TRAIN_OUTPUT_DIR ${TRAIN_OUTPUT_DIR}"
echo "============================================================="

##############Datat Preprocessing################
#To hkl data
#if [ -d "$DATA_PREPROCESS_DIR" ]; then
#    echo "The Preprocessed Data (.hkl ) exist"
#else
#    python ../workflow_video_prediction/DataPreprocess/benchmark/mpi_stager_v2_process_netCDF.py \
#    --input_dir ${DATA_EXTRA_DIR} --destination_dir ${DATA_PREPROCESS_DIR}
#fi

####Change the .hkl data to .tfrecords files
if [ -d "$DATA_PREPROCESS_TF_DIR" ]
then
    echo "Step2: The Preprocessed Data (tf.records) exist"
else
    echo "Step2: start, hkl. files to tf.records"
    python ./video_prediction/datasets/era5_dataset_v2.py  --source_dir ${DATA_PREPROCESS_DIR}/splits \
    --destination_dir ${DATA_PREPROCESS_TF_DIR}
    echo "Step2: finish"
fi

#########Train##########################
if [ "$TRAIN_MODE" == "pre_trained" ]; then
    echo "step3: Using kth pre_trained model"
elif [ "$TRAIN_MODE" == "end_to_end" ]; then
    echo "step3: End-to-end training"
    if [ "$RETRAIN" == 1 ]; then
        echo "Using the existing end-to-end model"
    else
        echo "Training Starts "
        python ./scripts/train_v2.py --input_dir $DATA_PREPROCESS_TF_DIR --dataset era5  \
        --model ${MODEL} --model_hparams_dict hparams/kth/${method_dir}/model_hparams.json \
        --output_dir ${TRAIN_OUTPUT_DIR} --checkpoint ${CHECKPOINT_DIR}
        echo "Training ends "
    fi
else
    echo "TRAIN_MODE is end_to_end or pre_trained"
    exit 1
fi

#########Generate results#################
echo "Step4: Postprocessing start"
python ./scripts/generate_transfer_learning_finetune.py --input_dir ${DATA_PREPROCESS_TF_DIR} \
--dataset_hparams sequence_length=20 --checkpoint ${CHECKPOINT_DIR} --mode test --results_dir ${RESULTS_OUTPUT_DIR} \
--batch_size 4 --dataset era5
