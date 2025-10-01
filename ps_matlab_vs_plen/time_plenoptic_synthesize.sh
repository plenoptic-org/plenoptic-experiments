#!/usr/bin/env bash

start=$(date '+%s.%N')
PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic OMP_NUM_THREADS=1 python ~/plenoptic_experiments/ps_matlab_vs_plen/synthesize.py --img $1 --init_img $2 --optimizer $3 --history_size $4 --device $5 --synth_max_iter $6 --weighted $7 --output_path $8
stop=$(date '+%s.%N')
elapsed=$(bc -l <<< "$stop - $start")
echo $elapsed
