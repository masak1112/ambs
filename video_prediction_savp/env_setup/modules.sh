#!/usr/bin/env bash


module purge
module use $OTHERSTAGES
module load Stages/2019a
module load GCC/8.3.0
if [[ "${HOST_NAME}" == hdfml* ]]; then
  # MVAPICH2 conflicts with h5py (since h5py requires MPIParaStation)
  module load MVAPICH2/2.3.3-GDR
elif [[ "${HOST_NAME}" == juwels* ]]; then
  module load MVAPICH2/.2.3.1-GDR
fi
module load GCCcore/.8.3.0
module load mpi4py/3.0.1-Python-3.6.8
#module load h5py/2.9.0-serial-Python-3.6.8
module load h5py/2.9.0-Python-3.6.8
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load cuDNN/7.5.1.10-CUDA-10.1.105

