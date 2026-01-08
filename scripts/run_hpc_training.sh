#!/bin/bash
# HPC Training Execution Script
# Usage: bash scripts/run_hpc_training.sh OA_Screening

TASK=${1:-all}

# HPC paths
export DATA_PATH=/home2/gun3856/gait_analysis/dataset/data
export OUTPUT_PATH=/home2/gun3856/gait_code/results
export MODEL_PATH=/home2/gun3856/gait_code/models

# Create directories
mkdir -p logs results models

# Run training
echo "Starting training for task: $TASK"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"

nohup python -u src/train_baseline_hpc.py --task $TASK > logs/${TASK}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "Started with PID: $PID"
echo $PID > ${TASK}_pid.txt

echo ""
echo "Monitor with: tail -f logs/${TASK}_*.log"
echo "Check process: ps aux | grep $PID"
echo "Kill if needed: kill $PID"
