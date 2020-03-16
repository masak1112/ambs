module --force purge
module /usr/local/software/jureca/OtherStages
module load Stages/2019a
module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1
module load mpi4py/3.0.1-Python-3.6.8
pip3 install --user netCDF4
pip3 install --user numpy


