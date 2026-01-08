#!/bin/bash

# HPC 결과 파일 다운로드 스크립트
# 사용법: bash scripts/download_hpc_results.sh

HPC_USER="gun3856"
HPC_HOST="hpc.yonsei.ac.kr"
HPC_RESULTS_DIR="~/gait_analysis/results"
HPC_MODELS_DIR="~/gait_analysis/models"
LOCAL_RESULTS_DIR="D:/gait_wearable_sensor/results"
LOCAL_MODELS_DIR="D:/gait_wearable_sensor/models"

echo "=== HPC 결과 파일 다운로드 ==="
echo ""

# Results 디렉토리 생성 (없으면)
mkdir -p "$LOCAL_RESULTS_DIR"
mkdir -p "$LOCAL_MODELS_DIR"

# CSV 결과 파일 다운로드
echo "1. CSV 결과 파일 다운로드 중..."
scp ${HPC_USER}@${HPC_HOST}:${HPC_RESULTS_DIR}/dl_baseline_results_*.csv "$LOCAL_RESULTS_DIR/"

if [ $? -eq 0 ]; then
    echo "✅ CSV 파일 다운로드 완료"
else
    echo "❌ CSV 파일 다운로드 실패"
fi

echo ""

# 모델 파일 다운로드 (선택사항 - 파일 크기 큼)
read -p "모델 파일(.pt)도 다운로드하시겠습니까? (y/n): " download_models

if [ "$download_models" = "y" ]; then
    echo "2. 모델 파일 다운로드 중..."
    scp ${HPC_USER}@${HPC_HOST}:${HPC_MODELS_DIR}/*_best.pt "$LOCAL_MODELS_DIR/"

    if [ $? -eq 0 ]; then
        echo "✅ 모델 파일 다운로드 완료"
    else
        echo "❌ 모델 파일 다운로드 실패"
    fi
else
    echo "모델 파일 다운로드 건너뜀"
fi

echo ""
echo "=== 다운로드 완료 ==="
echo "결과 파일 위치: $LOCAL_RESULTS_DIR"
echo "모델 파일 위치: $LOCAL_MODELS_DIR"

# 다운로드된 파일 목록 출력
echo ""
echo "다운로드된 CSV 파일:"
ls -lh "$LOCAL_RESULTS_DIR"/dl_baseline_results_*.csv 2>/dev/null || echo "없음"

echo ""
echo "다운로드된 모델 파일:"
ls -lh "$LOCAL_MODELS_DIR"/*_best.pt 2>/dev/null || echo "없음"
