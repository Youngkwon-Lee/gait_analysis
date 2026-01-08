#!/bin/bash
# Feature Importance Analysis Script for HPC
# Usage: bash scripts/run_feature_importance.sh [task_name]

TASK=${1:-OA_Screening}

# HPC paths
export DATA_PATH=/home2/gun3856/gait_analysis/dataset/data
export OUTPUT_PATH=/home2/gun3856/gait_code/results
export MODEL_PATH=/home2/gun3856/gait_code/models

# Create directories
mkdir -p results/feature_importance

echo "================================"
echo "Feature Importance Analysis"
echo "Task: $TASK"
echo "================================"

# Run analysis
nohup python -u src/analyze_feature_importance.py --task $TASK > logs/feature_importance_${TASK}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "Started with PID: $PID"
echo $PID > feature_importance_pid.txt

echo ""
echo "Monitor with: tail -f logs/feature_importance_${TASK}_*.log"
echo "Check process: ps aux | grep $PID"
echo "Kill if needed: kill $PID"
