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

set -euo pipefail
mkdir -p logs
source .venv/bin/activate
nvidia-smi
python scripts/s04_train.py --config "$1"
