#!/usr/bin/env bash

start=$(date '+%s.%N')
PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic OMP_NUM_THREADS=1 python ~/plenoptic_experiments/ps_noise/synthesize.py --img $1 --init_noise $2 --init_noise_type $3 --init_seed $4 --history_size $5 --device $6 --synth_max_iter $7 --weighted $8 --output_path $9
stop=$(date '+%s.%N')
elapsed=$(bc -l <<< "$stop - $start")
echo $elapsed
