#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=comp         # job name (shown in squeue and notifications)
#SBATCH --output=%x.%j.out          # stdout file (%x = job name, %j = job ID)
#SBATCH --error=%x.%j.err           # stderr file
#SBATCH --time=00:30:00             # wall-clock time limit (HH:MM:SS)
#SBATCH --partition=GPU-shared             # partition to submit to
#SBATCH --gres=gpu:v100-32:4        # select gpu type

pwd 
rm -r build

module load cuda

cmake -B build -DPROBLEM=fluids/turb -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper
#cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DAthena_ENABLE_MPI=On -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON


cmake --build build -j
