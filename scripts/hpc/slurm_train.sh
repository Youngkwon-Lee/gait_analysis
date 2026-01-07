#!/bin/bash
#SBATCH --job-name=gait_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# SLURM batch script for gait classification
# Submit: sbatch slurm_train.sh

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "Started: $(date)"

# Load modules (adjust for your HPC)
module load cuda/11.8
module load cudnn/8.6

# Activate environment
source ~/.bashrc
conda activate gait

# Set paths
export DATA_PATH=~/gait_analysis/dataset/data
export OUTPUT_PATH=~/gait_analysis/results
export MODEL_PATH=~/gait_analysis/models

# GPU info
nvidia-smi

# Run training
cd ~/gait_analysis
python -u src/train_baseline_hpc.py --task all --gpu 0

echo "Finished: $(date)"
