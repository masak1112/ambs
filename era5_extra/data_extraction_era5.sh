#!/bin/bash -x
#SBATCH --account-deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --partition=batch

# Load modules...
ml GCC/9.3.0 ParaStationMPI/5.4.7-1 CDO/1.9.8

#source_dir="/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib"
#destination_dir="/p/home/jusers/ji4/juwels/ambs/era5_extra"

#year=2010
#var1='2t' 
#var2='t850'

#$run python extract_era5.py --target_year $year --target_variable $var1 $var2 
