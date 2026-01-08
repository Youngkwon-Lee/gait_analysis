# Phase 1 분석 HPC 실행 가이드

**Phase 1 분석 항목**:
1. Error Analysis (에러 분석) - 틀린 케이스 분석
2. Confusion Analysis (혼동 분석) - False Positive/Negative 비교 및 임계값 최적화

---

## 🚀 빠른 실행 (원라이너)

### 1단계: 파일 업로드
```bash
scp src/analyze_errors.py src/analyze_confusion.py kesl:/scratch/x2026a01/gait_wearable_sensor/src/
```

### 2단계: Error Analysis 실행
```bash
ssh kesl "cd /scratch/x2026a01/gait_wearable_sensor && export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data && export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/error_analysis && export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models && nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &"
```

### 3단계: Confusion Analysis 실행 (Error Analysis 완료 후)
```bash
ssh kesl "cd /scratch/x2026a01/gait_wearable_sensor && export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data && export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis && export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models && nohup python -u src/analyze_confusion.py > logs/confusion_analysis.log 2>&1 &"
```

### 4단계: 로그 확인 (1분 후)
```bash
# Error Analysis 로그
ssh kesl "tail -50 /scratch/x2026a01/gait_wearable_sensor/logs/error_analysis.log"

# Confusion Analysis 로그
ssh kesl "tail -50 /scratch/x2026a01/gait_wearable_sensor/logs/confusion_analysis.log"
```

### 5단계: 결과 다운로드 (완료 후)
```bash
# Error Analysis 결과
scp -r kesl:/scratch/x2026a01/gait_wearable_sensor/results/error_analysis/* D:/gait_wearable_sensor/results/error_analysis/

# Confusion Analysis 결과
scp -r kesl:/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis/* D:/gait_wearable_sensor/results/confusion_analysis/
```

---

## 📋 단계별 상세 가이드

### Step 1: 파일 업로드 및 디렉토리 확인

```bash
# 1. 로컬 파일 HPC로 업로드
scp src/analyze_errors.py kesl:/scratch/x2026a01/gait_wearable_sensor/src/
scp src/analyze_confusion.py kesl:/scratch/x2026a01/gait_wearable_sensor/src/

# 2. HPC 접속
ssh kesl

# 3. 디렉토리 확인
cd /scratch/x2026a01/gait_wearable_sensor
ls -la src/analyze_*.py
ls -la models/OA_Screening_best.pth  # 모델 파일 확인
```

### Step 2: Error Analysis 실행

```bash
# HPC에서 실행
cd /scratch/x2026a01/gait_wearable_sensor

# 환경변수 설정 및 실행
export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/error_analysis
export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models

# 백그라운드 실행
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &

# 프로세스 확인
ps aux | grep analyze_errors

# 로그 실시간 모니터링
tail -f logs/error_analysis.log
```

**예상 실행 시간**: 5-10분

### Step 3: Confusion Analysis 실행

Error Analysis 완료 후 실행:

```bash
# 환경변수 설정 및 실행
export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis
export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models

# 백그라운드 실행
nohup python -u src/analyze_confusion.py > logs/confusion_analysis.log 2>&1 &

# 로그 실시간 모니터링
tail -f logs/confusion_analysis.log
```

**예상 실행 시간**: 5-10분

### Step 4: 결과 확인

```bash
# HPC에서 결과 파일 확인
ls -lh /scratch/x2026a01/gait_wearable_sensor/results/error_analysis/
ls -lh /scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis/

# 로그 전체 보기
cat logs/error_analysis.log
cat logs/confusion_analysis.log
```

### Step 5: 결과 다운로드

```bash
# 로컬로 돌아와서 실행 (HPC에서 exit)
exit

# Error Analysis 결과 다운로드
mkdir -p D:/gait_wearable_sensor/results/error_analysis
scp -r kesl:/scratch/x2026a01/gait_wearable_sensor/results/error_analysis/* \
    D:/gait_wearable_sensor/results/error_analysis/

# Confusion Analysis 결과 다운로드
mkdir -p D:/gait_wearable_sensor/results/confusion_analysis
scp -r kesl:/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis/* \
    D:/gait_wearable_sensor/results/confusion_analysis/

# 로그도 다운로드
scp kesl:/scratch/x2026a01/gait_wearable_sensor/logs/error_analysis.log \
    D:/gait_wearable_sensor/logs/
scp kesl:/scratch/x2026a01/gait_wearable_sensor/logs/confusion_analysis.log \
    D:/gait_wearable_sensor/logs/
```

---

## 📊 예상 결과 파일

### Error Analysis 출력
```
results/error_analysis/
├── OA_Screening_error_analysis.json    # 에러 분석 JSON
└── OA_Screening_error_analysis.png     # 시각화 (7개 subplot)
    ├── Confusion Matrix
    ├── ROC Curve
    ├── Precision-Recall Curve
    ├── Probability Distribution
    ├── Error Type Comparison
    ├── Probability Box Plot
    └── Summary Statistics
```

### Confusion Analysis 출력
```
results/confusion_analysis/
├── OA_Screening_confusion_analysis.json    # 임계값 분석 JSON
└── OA_Screening_confusion_analysis.png     # 시각화 (10+ subplot)
    ├── ROC Curve with Optimal Points
    ├── Precision-Recall Curve
    ├── Sensitivity vs Specificity
    ├── Youden's Index
    ├── F1 Score vs Threshold
    ├── Error Counts vs Threshold
    ├── Confusion Matrices (4 different thresholds)
    └── Clinical Recommendations
```

---

## 🔍 예상 출력 (로그)

### Error Analysis 로그
```
================================================================================
Error Analysis - Phase 1-1
================================================================================

Loading model from /scratch/.../models/OA_Screening_best.pth...
Model loaded successfully

Loading data...
Loaded Class 0 (HS): 360 trials
Loaded Class 1 (OA): 152 trials

Train: 410 trials, 85 subjects
Test: 102 trials, 21 subjects
Test dataset: ~1020 windows

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

### Confusion Analysis 로그
```
================================================================================
Confusion Analysis - Phase 1-2
================================================================================

Loading model...
Model loaded successfully

Loading data...
Test dataset: ~1020 windows

Getting predictions...

[1/3] Finding optimal thresholds...

[2/3] Analyzing threshold impact...

================================================================================
OPTIMAL THRESHOLD ANALYSIS
================================================================================

DEFAULT (Threshold: 0.500)
  Reason: Standard classification threshold
  Sensitivity (Recall): 0.xxxx - xx/xx OA patients detected
  Specificity: 0.xxxx - xxx/xxx healthy correctly identified
  PPV (Precision): 0.xxxx - xx/xx positive predictions correct
  NPV: 0.xxxx - xxx/xxx negative predictions correct
  Confusion: TN=xxx, FP=xx, FN=xx, TP=xx

YOUDEN (Threshold: 0.xxx)
  Reason: Maximizes (Sensitivity + Specificity - 1)
  Sensitivity: 0.xxxx
  Specificity: 0.xxxx
  ...

HIGH_SENSITIVITY (Threshold: 0.xxx)
  Reason: Maintains ≥95% sensitivity (catch most OA patients)
  Sensitivity: 0.95xx
  Specificity: 0.xxxx
  ...

HIGH_SPECIFICITY (Threshold: 0.xxx)
  Reason: Maintains ≥95% specificity (minimize false alarms)
  Sensitivity: 0.xxxx
  Specificity: 0.95xx
  ...

[OK] Results saved: OA_Screening_confusion_analysis.json
[3/3] Creating visualizations...
[OK] Visualization saved: OA_Screening_confusion_analysis.png

[DONE] Confusion Analysis Complete!
```

---

## 🛠️ 문제 해결

### 모델 파일이 없을 경우
```bash
# 모델 확인
ssh kesl "ls -la /scratch/x2026a01/gait_wearable_sensor/models/"

# 없으면 baseline 학습 먼저
ssh kesl "cd /scratch/x2026a01/gait_wearable_sensor && \
  nohup python -u src/train_baseline_hpc.py --task OA_Screening > logs/oa.log 2>&1 &"
```

### seaborn 모듈 없을 경우
```bash
ssh kesl
pip install --user seaborn
```

### 권한 오류
```bash
# 디렉토리 생성
ssh kesl "mkdir -p /scratch/x2026a01/gait_wearable_sensor/results/error_analysis"
ssh kesl "mkdir -p /scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis"
ssh kesl "mkdir -p /scratch/x2026a01/gait_wearable_sensor/logs"
```

### 프로세스 확인 및 종료
```bash
# 실행 중인 프로세스 확인
ssh kesl "ps aux | grep analyze"

# 종료 (필요시)
ssh kesl "pkill -f analyze_errors"
ssh kesl "pkill -f analyze_confusion"
```

---

## 📈 결과 해석 가이드

### Error Analysis 결과 보는 법
1. **Confusion Matrix**: 어디서 틀렸는지 한눈에 확인
2. **False Positive vs False Negative**: 어느 에러가 더 많은지
3. **Probability Distribution**: 에러의 확신도 (확실하게 틀렸는지, 애매하게 틀렸는지)
4. **개선 방향**: FN이 많으면 민감도 높이기, FP가 많으면 정밀도 높이기

### Confusion Analysis 결과 보는 법
1. **Optimal Threshold**: 목적에 따라 다른 임계값 선택
   - 스크리닝: High Sensitivity (환자 놓치지 않기)
   - 확진: High Specificity (정확한 진단)
   - 연구: Youden's Index (균형)

2. **Trade-off 이해**:
   - Threshold ↓ → 더 많은 환자 잡음 (FN↓) but 오진 증가 (FP↑)
   - Threshold ↑ → 정확한 진단 (FP↓) but 환자 놓침 (FN↑)

3. **임상 적용**:
   - 초기 스크리닝: 0.3-0.4 (높은 민감도)
   - 최종 진단: 0.6-0.7 (높은 특이도)

---

## ✅ 체크리스트

- [ ] analyze_errors.py 업로드 완료
- [ ] analyze_confusion.py 업로드 완료
- [ ] Error Analysis 실행 완료 (logs/error_analysis.log 확인)
- [ ] Confusion Analysis 실행 완료 (logs/confusion_analysis.log 확인)
- [ ] Error Analysis 결과 다운로드
- [ ] Confusion Analysis 결과 다운로드
- [ ] 시각화 파일 확인 (.png)
- [ ] JSON 결과 파일 확인

---

**다음 단계**: Phase 2 분석 (시간 패턴 분석, 질환 심각도 분석)
