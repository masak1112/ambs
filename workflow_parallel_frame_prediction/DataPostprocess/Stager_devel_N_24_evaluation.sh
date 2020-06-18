#!/bin/bash
#SBATCH --account=jjsc42
# budget account where contingent is taken from# TASKS = NODES * GPUS_PER_NODE
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=12
# can be omitted if --nodes and --ntasks-per-node
# are given
# SBATCH --cpus-per-task=1
# for OpenMP/hybrid jobs only
#SBATCH --output=horovod-4ntasks%j.out
# if keyword omitted: Default is slurm-%j.out in
# the submission directory (%j is replaced by
# the job ID).
#SBATCH --error=horovod-4ntasks%j.err
# if keyword omitted: Default is slurm-%j.out in
# the submission directory.
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpus
#SBATCH --mail-user=b.gong@fz-juelich.de
#SBATCH --mail-type=ALL

#create a folder to save the output

module --force purge
module --force  purge
module load Stages/Devel-2019a
module load GCC/8.3.0
module load MVAPICH2/2.3.2-GDR
#module /usr/local/software/jureca/OtherStages
module load Stages/2019a
module load GCCcore/.8.3.0
module load cuDNN/7.5.1.10-CUDA-10.1.105
module load Horovod/0.16.2-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8
#module load Intel/2019.3.199-GCC-8.3.0  ParaStationMPI/5.2.2-1-mt
#module load mpi4py/3.0.1-Python-3.6.8

srun python3.6 kitti_evaluate_parallel.py
