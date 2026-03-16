#!/bin/bash
#SBATCH --output=/home/jsobolma/rtx_queuer/slurm-%j.out

cd ~/rtx_queuer
export PYTHONUNBUFFERED=1
~/.local/bin/uv run --extra gpu python -u scripts/gpu_placeholder.py
