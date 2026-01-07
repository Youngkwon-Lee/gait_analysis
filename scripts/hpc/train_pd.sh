#!/bin/bash
# Train PD Screening task only
# Usage: nohup bash train_pd.sh > logs/pd_$(date +%Y%m%d_%H%M%S).log 2>&1 &

source ~/.bashrc
conda activate gait

export DATA_PATH=~/gait_analysis/dataset/data
export OUTPUT_PATH=~/gait_analysis/results
export MODEL_PATH=~/gait_analysis/models

python -u src/train_baseline_hpc.py --task PD_Screening --gpu 0
