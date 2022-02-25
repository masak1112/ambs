#!/bin/bash -x

#User's input : your virtual enviornment name
VIRT_ENV_NAME=venv_test

echo "Activating virtual environment..."
source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate

#checkpoint_dir: the checkpoint directory from train step
checkpoint_dir=/path/to/checkpoint/directory
#Results_dir: the output dir to save the results
results_dir=/path/to/results/directory
#Climate_file: the netcdf file point to the climtology, which you can download along with the samples data
climate_file=/home/b.gong/data_era5/T2monthly/climatology_t2m_1991-2020.nc
#select models
model=convLSTM

# The --lquick_evaluation enable you to quick test and generate the results. If you want to test on full large test dataset, please remove the --lquick_evaluation argument
python3 ../main_scripts/main_visualize_postprocess.py --checkpoint  ${checkpoint_dir} --mode test  \
                                                           --results_dir ${results_dir} --batch_size 4 \
                                                           --num_stochastic_samples 1 \
							   --lquick_evaluation --climatology_file ${climate_file}

