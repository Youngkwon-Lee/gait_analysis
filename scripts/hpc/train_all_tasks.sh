#!/bin/bash
# Train all classification tasks on HPC
# Usage: nohup bash train_all_tasks.sh > logs/train_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "=========================================="
echo "Gait Classification - All Tasks Training"
echo "Started: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate gait

# Set paths (adjust for your HPC)
export DATA_PATH=~/gait_analysis/dataset/data
export OUTPUT_PATH=~/gait_analysis/results
export MODEL_PATH=~/gait_analysis/models

# Create directories
mkdir -p $OUTPUT_PATH
mkdir -p $MODEL_PATH
mkdir -p ~/gait_analysis/logs

# GPU check
nvidia-smi

# Train all tasks
python -u src/train_baseline_hpc.py --task all --gpu 0

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
