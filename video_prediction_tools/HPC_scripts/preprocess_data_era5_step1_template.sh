#!/bin/bash -x
## Controlling Batch-job
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --output=DataPreprocess_era5_step1-out.%j
#SBATCH --error=DataPreprocess_era5_step1-err.%j
#SBATCH --time=00:20:00
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

######### Template identifier (don't remove) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (don't remove) #########

# Name of virtual environment 
VIRT_ENV_NAME="my_venv"

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


# select years and variables for dataset and define target domain 
years=( "2015" )
variables=( "t2" "t2" "t2" )
lat_inds=( 74 202 )
lon_inds=( 550 710 )

max_year=$( printf "%d\n" "${years[@]}" | sort -n | tail -1 )
min_year=$( printf "%d\n" "${years[@]}" | sort -nr | tail -1 )
# set some paths
# note, that destination_dir is used during runtime to set a proper experiment directory
source_dir=/p/scratch/deepacf/video_prediction_shared_folder/extractedData
destination_dir=/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y${min_year}to${max_year}M01to12
script_dir=`pwd`

# execute Python-scripts
for year in "${years[@]}";     do 
        echo "Year $year"
	echo "source_dir ${source_dir}/${year}"
	srun python ../main_scripts/main_preprocess_data_step1.py \
        --source_dir ${source_dir} -scr_dir ${script_dir} -exp_id ${exp_id} \
        --destination_dir ${destination_dir} --years ${year} --vars ${variables[0]} ${variables[1]} ${variables[2]}
       	--lat_s ${lat_inds[0]} --lat_e ${lat_inds[1]} --lon_s ${lon_inds[0]} --lon_e ${lon_inds[1]}    
    done


#srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500    
