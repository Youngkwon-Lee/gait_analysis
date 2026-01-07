# HPC Environment Setup for Gait Analysis

## Overview

HPC 환경에서 Deep Learning 모델 학습을 위한 설정 가이드입니다.

## HPC 접속 정보

```bash
# SSH 접속 (Hawkeye 프로젝트 참고)
ssh username@hpc.server.address

# 작업 디렉토리
cd /path/to/gait_analysis
```

## 파일 전송

HPC 환경에서는 git이 작동하지 않으므로 **wget**을 사용합니다:

```bash
# 로컬에서 파일 서버로 업로드 후
wget http://your-server/gait_analysis.tar.gz
tar -xzf gait_analysis.tar.gz

# 또는 scp 사용
scp -r D:\gait_wearable_sensor\src username@hpc:/path/to/gait_analysis/
scp -r D:\gait_wearable_sensor\dataset username@hpc:/path/to/gait_analysis/
```

## GPU 환경

- **사용 가능 GPU**: V100, A100
- **메모리**: 16GB-80GB (모델에 따라 선택)

```bash
# GPU 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version
```

## Python 환경 설정

```bash
# Conda 환경 생성
conda create -n gait python=3.10
conda activate gait

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기타 dependencies
pip install -r requirements-hpc.txt
```

### requirements-hpc.txt

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
tensorboard>=2.13.0
wandb>=0.15.0
einops>=0.6.0
```

## 학습 실행

### 백그라운드 실행 (nohup)

```bash
# 학습 시작
nohup python -u train_baseline.py --config configs/pd_screening.yaml > logs/pd_screening.log 2>&1 &

# 프로세스 확인
ps aux | grep python

# 로그 모니터링
tail -f logs/pd_screening.log

# 프로세스 종료 (필요시)
kill -9 <PID>
```

### SLURM 사용 (클러스터 환경)

```bash
#!/bin/bash
#SBATCH --job-name=gait_pd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source activate gait
python train_baseline.py --task pd_screening
```

```bash
# 작업 제출
sbatch scripts/train_pd.sh

# 작업 상태 확인
squeue -u $USER

# 작업 취소
scancel <JOB_ID>
```

## 학습 설정

### 4가지 Binary Classification Tasks

| Task | Class 0 | Class 1 | 예상 학습 시간 |
|------|---------|---------|---------------|
| PD_Screening | HS (360) | PD (160) | ~2시간 |
| OA_Screening | HS (360) | HOA (74) | ~1.5시간 |
| CVA_Detection | HS (360) | CVA (128) | ~2시간 |
| PD_vs_CVA | PD (160) | CVA (128) | ~1.5시간 |

### 모델 설정

```yaml
# configs/default.yaml
model:
  type: MultiStreamAttentionCNN
  input_channels: 9  # Acc(3) + Gyr(3) + Mag(3), or 6 without Mag
  hidden_dim: 64
  num_heads: 4
  dropout: 0.3

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine

data:
  window_size: 300  # 3 seconds at 100Hz
  stride: 150       # 50% overlap
  exclude_mag: true # Magnetometer 제외 옵션
```

## 결과 확인

```bash
# 학습 결과 다운로드
scp username@hpc:/path/to/gait_analysis/results/* ./results/

# TensorBoard
tensorboard --logdir=logs/tensorboard

# wandb (온라인)
# https://wandb.ai/your-project/gait-analysis
```

## 트러블슈팅

### CUDA Out of Memory

```python
# Batch size 줄이기
batch_size = 16  # from 32

# Gradient accumulation 사용
accumulation_steps = 2
```

### 학습이 멈춤

```bash
# GPU 상태 확인
watch -n 1 nvidia-smi

# 프로세스 확인
ps aux | grep python | grep -v grep
```

### 로그가 안 나옴

```bash
# Python unbuffered 모드
python -u train.py

# 또는 코드에서
import sys
sys.stdout.reconfigure(line_buffering=True)
```

## Baseline Paper 목표 성능

| Task | ROC-AUC | Balanced Acc |
|------|---------|--------------|
| PD_Screening | 0.821 | 0.639 |
| OA_Screening | 0.990 | 0.942 |
| CVA_Detection | 0.950 | 0.747 |
| PD_vs_CVA | 0.657 | 0.607 |

## 현재 RF Baseline 결과 (No Magnetometer)

| Task | ROC-AUC | Balanced Acc |
|------|---------|--------------|
| PD_Screening | 0.998 | 0.857 |
| OA_Screening | 0.998 | 0.985 |
| CVA_Detection | 1.000 | 1.000 |
| PD_vs_CVA | 0.912 | 0.852 |

**Note**: RF 결과가 매우 높음 - Deep Learning으로 검증 필요

## Quick Start

```bash
# 1. HPC 접속
ssh hpc

# 2. 환경 활성화
conda activate gait

# 3. 학습 시작
cd /path/to/gait_analysis
nohup python -u train_baseline.py --task all > logs/train_all.log 2>&1 &

# 4. 로그 모니터링
tail -f logs/train_all.log
```
