#!/bin/bash -x

#User's input : your virtual enviornment name
VIRT_ENV_NAME=venv_test

echo "Activating virtual environment..."
source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate

#the source directory contains the tfrecords
checkpoint_dir=/home/b.gong/model/checkpoint_89
results_dir=/home/b.gong/results/
lquick=1
climate_file=/home/b.gong/data_era5/T2monthly/climatology_t2m_1991-2020.nc
#select models
model=convLSTM
#mkdir ${results_dir}
python3 ./video_prediction_tools/main_scripts/main_visualize_postprocess.py --checkpoint  ${checkpoint_dir} --mode test  \
                                                           --results_dir ${results_dir} --batch_size 4 \
                                                           --num_stochastic_samples 1 \
							   --lquick_evaluation ${lquick} --climatology_file ${climate_file}

