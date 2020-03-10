#!/usr/bin/env bash
set -e

model="savp"
#train_mode: end_to_end, pre_trained, fine_tune
train_mode="end_to_end"
is_pretrain=False
exp_name="era5_size_64_64_3_3t_norm"

if [ $model=="savp" ]
then
    method_dir="ours_savp"
elif [ $model=="gan" ]
then
    method_dir="ours_gan"
elif [ $model=="vae" ]
then
    method_dir="ours_vae"
fi

raw_dataset_input=./splits/${exp_name}
prep_data_input=./data/${exp_name}
train_output=./logs/${exp_name}
results_output=./results_test_samples/${exp_name}/${method_dir}


##############Datat Preprocessing################
python ./video_prediction/datasets/era5_dataset_v2.py  ${raw_dataset_input}  ${prep_data_input}

#########Train##########################
python ./scripts/train_v2.py --input_dir ${prep_data_input} --dataset era5  \
--model savp --model_hparams_dict hparams/kth/${method_dir}/model_hparams.json \
--output_dir logs/era5_64_64_3_3t_norm/${train_mode}/${method_dir}\

#--checkpoint pretrained_models/kth/ours_savp
#########Generate results#################
python ./scripts/generate_transfer_learning_finetune.py --input_dir ${prep_data_input} \
--dataset_hparams sequence_length=20 --checkpoint ${train_output}\
--mode test --results_dir ${results_output} \
--batch_size 4 --dataset era5









