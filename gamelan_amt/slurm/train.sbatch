#!/bin/bash
# Usage: sbatch slurm/train.sbatch configs/lora_cents.yaml
# Re-submit the same line to resume after the 24-h limit (s04_train.py
# saves <out_dir>/last and stops itself at max_hours).
#SBATCH --job-name=gamelan-lora
#SBATCH --gres=gpu:1            # ask your admin for the exact MIG/GPU syntax
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300    # warn training 5 min before the time limit so it saves

set -euo pipefail
mkdir -p logs
source .venv/bin/activate
nvidia-smi
# exec so python itself receives SIGUSR1 / SIGTERM (scancel) and saves last/
exec python scripts/s04_train.py --config "$1"
