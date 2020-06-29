#!/usr/bin/env bash

sed -i "s|source_dir=.*|source_dir=${SAVE_DIR}preprocessedData/|g" DataPreprocess_to_tf.sh
sed -i "s|destination_dir=.*|destination_dir=${SAVE_DIR}preprocessedData/|g" DataPreprocess_to_tf.sh

sed -i "s|source_dir=.*|source_dir=${SAVE_DIR}preprocessedData/|g" train_era5.sh
sed -i "s|destination_dir=.*|destination_dir=${SAVE_DIR}models/|g" train_era5.sh

sed -i "s|source_dir=.*|source_dir=${SAVE_DIR}preprocessedData/|g" generate_era5.sh
sed -i "s|checkpoint_dir=.*|checkpoint_dir=${SAVE_DIR}models/|g" generate_era5.sh
sed -i "s|results_dir=.*|results_dir=${SAVE_DIR}results/|g" generate_era5.sh
