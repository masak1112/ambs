#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=<your_project>
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --output=meta_postprocess_era5-out.%j
#SBATCH --error=meta_postprocess_era5-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=me@somewhere.com

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Declare input parameters
root_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/
analysis_config=video_prediction_tools/meta_postprocess_config/meta_config.json
metric=mse
exp_id=test
enable_skill_scores=True

srun python ../main_scripts/main_meta_postprocess.py  --root_dir ${root_dir} --analysis_config ${analysis_config} \
                                                       --metric ${metric} --exp_id ${exp_id} --enable_skill_scores ${enable_skill_scores}
