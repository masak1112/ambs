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

max_year=`echo "${years[*]}" | sort -nr | head -n1`
min_year=`echo "${years[*]}" | sort -nr | tail -n1`
# set some paths
# note, that destination_dir is used during runtime to set a proper experiment directory
exp_id=xxx                                          # experiment identifier is set by 'generate_workflow_runscripts.sh'
source_dir=${SAVE_DIR}/extractedData
destination_dir=${SAVE_DIR}/preprocessedData/era5-Y${min_year}to${max_year}M01to12
script_dir=`pwd`

# execute Python-scripts
for year in "${years[@]}";     do 
        echo "Year $year"
	echo "source_dir ${source_dir}/${year}"
	srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
        --source_dir ${source_dir} -scr_dir ${script_dir} -exp_id ${exp_id} \
        --destination_dir ${destination_dir} --years ${year} --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710     
    done


#srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500    
