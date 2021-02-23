#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess-out.%j
#SBATCH --error=DataPreprocess-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment 
VIRT_ENV_NAME="virt_env_hdfml"

# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ../${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ../${VIRT_ENV_NAME}/bin/activate
   else 
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi
# Loading mouldes
source ../env_setup/modules_preprocess.sh


# select years for dataset
declare -a years=(
                 "2015"
                 "2016"
                 "2017"
                  )

max_year=$( printf "%d\n" "${years[@]}" | sort -n | tail -1 )
min_year=$( printf "%d\n" "${years[@]}" | sort -nr | tail -1 )
# set some paths
# note, that destination_dir is used during runtime to set a proper experiment directory
exp_id=xxx                                          # experiment identifier is set by 'generate_workflow_runscripts.sh'
source_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/extractedData
destination_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y${min_year}to${max_year}M01to12
script_dir=`pwd`

# execute Python-scripts
for year in "${years[@]}";     do 
        echo "Year $year"
	echo "source_dir ${source_dir}/${year}"
	srun python ../main_scripts/main_preprocess_data_step1.py \
        --source_dir ${source_dir} -scr_dir ${script_dir} -exp_id ${exp_id} \
        --destination_dir ${destination_dir} --years ${year} --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710     
    done


#srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500    
