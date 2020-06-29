#!/bin/bash -x
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

module purge
module use  $OTHERSTAGES
module load Stages/2019a
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load h5py/2.9.0-Python-3.6.8
module load mpi4py/3.0.1-Python-3.6.8




source_dir=${SAVE_DIR}/extractedData
destination_dir=${SAVE_DIR}/preprocessedData/era5-Y2015to2017M01to12
script_dir=`pwd`

declare -a years=("2015" 
                 "2016" 
                 "2017"
                  )


for year in "${years[@]}"; 
    do 
        echo "Year $year"
	echo "source_dir ${source_dir}/${year}"
	srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py \
        --source_dir ${source_dir} -scr_dir ${script_dir} \
        --destination_dir ${destination_dir} --years ${year} --vars T2 MSL gph500 --lat_s 74 --lat_e 202 --lon_s 550 --lon_e 710     
    done


srun python ../../workflow_parallel_frame_prediction/DataPreprocess/mpi_split_data_multi_years.py --destination_dir ${destination_dir} --varnames T2 MSL gph500    
