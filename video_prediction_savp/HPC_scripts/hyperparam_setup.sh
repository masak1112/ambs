#!/usr/bin/env bash

# for choosing the model convLSTM vae mcnet savp
export model=convLSTM
export model_hparams=../hparams/era5/${model}/model_hparams.json

#create a subfolder with create time and user names, which can be consider as hyperparameter tunning folder. This can avoid overwrite the prevoius trained model using differ#ent hypermeters 
export hyperdir="$(date +"%Y%m%dT%H%M")_"$USER""

echo "model: ${model}"
echo "hparams: ${model_hparams}"
echo "experiment dir: ${hyperdir}"
