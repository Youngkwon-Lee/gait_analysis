# HPC에서 Error Analysis 실행하기

## 1. 파일 업로드
```bash
# 로컬 → HPC 업로드
scp src/analyze_errors.py kesl:/scratch/x2026a01/gait_wearable_sensor/src/
```

## 2. HPC에서 실행
```bash
# HPC 접속
ssh kesl

# 작업 디렉토리 이동
cd /scratch/x2026a01/gait_wearable_sensor

# 환경변수 설정
export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/error_analysis
export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models

# 백그라운드 실행
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &

# 로그 확인
tail -f logs/error_analysis.log
```

## 3. 결과 다운로드
```bash
# HPC → 로컬 다운로드
scp kesl:/scratch/x2026a01/gait_wearable_sensor/results/error_analysis/* \
    D:/gait_wearable_sensor/results/error_analysis/
```

## 4. 빠른 실행 (원라이너)
```bash
# 업로드
scp src/analyze_errors.py kesl:/scratch/x2026a01/gait_wearable_sensor/src/

# HPC 실행
ssh kesl "cd /scratch/x2026a01/gait_wearable_sensor && export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data && export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/error_analysis && export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models && nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &"

# 로그 확인 (30초 후)
sleep 30 && ssh kesl "tail -100 /scratch/x2026a01/gait_wearable_sensor/logs/error_analysis.log"
```

## 5. 예상 출력
```
================================================================================
Error Analysis - Phase 1-1
================================================================================

Loading model from /scratch/x2026a01/gait_wearable_sensor/models/OA_Screening_best.pth...
Model loaded successfully

Loading data...
Loaded Class 0 (HS): 360 trials
Loaded Class 1 (OA): 152 trials

Train: 410 trials, 85 subjects
Test: 102 trials, 21 subjects
Test dataset: 1020 windows

Getting predictions...

================================================================================
OVERALL PERFORMANCE
================================================================================
AUC: 0.9923
Balanced Accuracy: 0.9593

Confusion Matrix:
[[xxx xxx]
 [xxx xxx]]

TN: xxx, FP: xxx
FN: xxx, TP: xxx

Total Errors: xx/1020 (x.xx%)

False Positives (건강 → OA 오판): xx
  Mean probability: 0.xxxx
  Confidence range: [0.xxxx, 0.xxxx]
  Unique subjects: x

False Negatives (OA → 건강 오판): xx
  Mean probability: 0.xxxx
  Confidence range: [0.xxxx, 0.xxxx]
  Unique subjects: x

[1/3] Running error analysis...
[2/3] Saving results...
[OK] Results saved: OA_Screening_error_analysis.json

[3/3] Creating visualizations...
[OK] Visualization saved: OA_Screening_error_analysis.png

[DONE] Error Analysis Complete!
```

## 6. 문제 해결

### seaborn 없을 경우
```bash
ssh kesl
pip install --user seaborn
```

### 모델 파일 없을 경우
```bash
# 모델 파일 확인
ssh kesl "ls -la /scratch/x2026a01/gait_wearable_sensor/models/"

# 없으면 baseline 먼저 학습
ssh kesl "cd /scratch/x2026a01/gait_wearable_sensor && nohup python -u src/train_baseline_hpc.py --task OA_Screening > logs/oa.log 2>&1 &"
```

## 7. 생성될 파일
```
results/error_analysis/
├── OA_Screening_error_analysis.json    # 에러 분석 결과
└── OA_Screening_error_analysis.png     # 시각화 (7개 subplot)
```
