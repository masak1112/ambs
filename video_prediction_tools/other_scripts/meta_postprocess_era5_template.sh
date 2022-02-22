#!/bin/bash -x

#User's input : your virtual enviornment name
VIRT_ENV_NAME=venv_test

echo "Activating virtual environment..."
source ../virtual_envs/${VIRT_ENV_NAME}/bin/activate

#the source directory contains the tfrecords
root_dir=/home/b.gong/
analysis_config=../meta_postprocess_config/meta_config.json
vim ${analysis_config}
metric=mse
exp_id=test
#enable_skill_scores=""
python ../main_scripts/main_meta_postprocess.py  --root_dir ${root_dir} --analysis_config ${analysis_config} \
                                                       --metric ${metric} --exp_id ${exp_id} \
                                                       #--enable_skill_scores ${enable_skill_scores}

