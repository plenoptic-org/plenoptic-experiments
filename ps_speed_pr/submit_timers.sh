#!/bin/bash

#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH -o timers-%j.out
#SBATCH -e timers-%j.out

PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic OMP_NUM_THREADS=1 python timers.py $1
