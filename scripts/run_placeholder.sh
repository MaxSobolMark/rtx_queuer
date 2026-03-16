#!/bin/bash
#SBATCH --output=/home/jsobolma/rtx_queuer/slurm-%j.out

cd ~/rtx_queuer
uv run --extra gpu scripts/gpu_placeholder.py
