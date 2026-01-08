# Phase 1 분석 실행 가이드 (GitHub 방식)

**GitHub 저장소**: https://github.com/Youngkwon-Lee/gait_analysis

---

## 🚀 빠른 실행 (3단계)

### 1️⃣ HPC에서 최신 코드 받기
```bash
ssh kesl
cd /scratch/x2026a01/gait_wearable_sensor
git pull origin main
```

### 2️⃣ Phase 1 분석 실행 (백그라운드)
```bash
# Error Analysis
export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/error_analysis
export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &

# Confusion Analysis (동시 실행 가능)
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis
nohup python -u src/analyze_confusion.py > logs/confusion_analysis.log 2>&1 &
```

### 3️⃣ 로그 확인 (1-2분 후)
```bash
# Error Analysis 로그
tail -f logs/error_analysis.log

# Confusion Analysis 로그 (새 터미널)
tail -f logs/confusion_analysis.log
```

---

## 📋 상세 가이드

### Step 1: HPC 환경 확인

```bash
# HPC 접속
ssh kesl

# 작업 디렉토리 이동
cd /scratch/x2026a01/gait_wearable_sensor

# 현재 브랜치 확인
git branch

# 현재 상태 확인
git status
```

### Step 2: GitHub에서 최신 코드 가져오기

```bash
# 최신 코드 pull
git pull origin main

# 새로 추가된 파일 확인
ls -la src/analyze_*.py
ls -la *.md

# 파일 내용 확인 (선택)
head -20 src/analyze_errors.py
head -20 NEXT_ANALYSIS_PLAN.md
```

**Pull 후 확인**:
```
From https://github.com/Youngkwon-Lee/gait_analysis
 * branch            main       -> FETCH_HEAD
Updating 84b83b4..a3304a4
Fast-forward
 DATASET_DOCUMENTATION.md      | 225 ++++++++++++
 NEXT_ANALYSIS_PLAN.md         | 225 ++++++++++++
 RUN_ERROR_ANALYSIS_HPC.md     |  89 +++++
 RUN_PHASE1_HPC.md             | 356 +++++++++++++++++++
 src/analyze_confusion.py      | 598 ++++++++++++++++++++++++++++++++
 src/analyze_errors.py         | 515 ++++++++++++++++++++++++++++
 6 files changed, 2055 insertions(+)
```

### Step 3: 환경변수 설정 및 실행

```bash
# 필수 환경변수 설정
export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data
export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models

# 로그 디렉토리 확인
mkdir -p logs

# Error Analysis 실행
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/error_analysis
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &
echo "Error Analysis PID: $!"

# Confusion Analysis 실행
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis
nohup python -u src/analyze_confusion.py > logs/confusion_analysis.log 2>&1 &
echo "Confusion Analysis PID: $!"
```

**한 줄 명령어** (환경변수 + 실행):
```bash
# Error Analysis
cd /scratch/x2026a01/gait_wearable_sensor && \
export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data && \
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/error_analysis && \
export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models && \
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &

# Confusion Analysis
cd /scratch/x2026a01/gait_wearable_sensor && \
export DATA_PATH=/scratch/x2026a01/gait_wearable_sensor/dataset/data && \
export OUTPUT_PATH=/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis && \
export MODEL_PATH=/scratch/x2026a01/gait_wearable_sensor/models && \
nohup python -u src/analyze_confusion.py > logs/confusion_analysis.log 2>&1 &
```

### Step 4: 진행 상황 모니터링

```bash
# 프로세스 확인
ps aux | grep analyze

# 로그 실시간 확인 (Error Analysis)
tail -f logs/error_analysis.log

# 로그 실시간 확인 (Confusion Analysis)
tail -f logs/confusion_analysis.log

# 로그 마지막 50줄 보기
tail -50 logs/error_analysis.log
tail -50 logs/confusion_analysis.log

# 결과 파일 생성 확인
watch -n 5 "ls -lh results/error_analysis/ results/confusion_analysis/"
```

### Step 5: 완료 확인

```bash
# Error Analysis 완료 확인
tail -20 logs/error_analysis.log | grep "DONE"

# Confusion Analysis 완료 확인
tail -20 logs/confusion_analysis.log | grep "DONE"

# 생성된 파일 확인
ls -lh results/error_analysis/
ls -lh results/confusion_analysis/
```

**예상 출력**:
```
results/error_analysis/
total 1.5M
-rw-r--r-- 1 user group  2.5K Jan  8 14:30 OA_Screening_error_analysis.json
-rw-r--r-- 1 user group  1.5M Jan  8 14:30 OA_Screening_error_analysis.png

results/confusion_analysis/
total 2.0M
-rw-r--r-- 1 user group  3.2K Jan  8 14:32 OA_Screening_confusion_analysis.json
-rw-r--r-- 1 user group  2.0M Jan  8 14:32 OA_Screening_confusion_analysis.png
```

---

## 📥 로컬로 결과 다운로드

### 방법 1: scp 사용
```bash
# 로컬 터미널에서 실행

# Error Analysis 결과
mkdir -p D:/gait_wearable_sensor/results/error_analysis
scp -r kesl:/scratch/x2026a01/gait_wearable_sensor/results/error_analysis/* \
    D:/gait_wearable_sensor/results/error_analysis/

# Confusion Analysis 결과
mkdir -p D:/gait_wearable_sensor/results/confusion_analysis
scp -r kesl:/scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis/* \
    D:/gait_wearable_sensor/results/confusion_analysis/

# 로그도 다운로드
scp kesl:/scratch/x2026a01/gait_wearable_sensor/logs/error_analysis.log \
    D:/gait_wearable_sensor/logs/
scp kesl:/scratch/x2026a01/gait_wearable_sensor/logs/confusion_analysis.log \
    D:/gait_wearable_sensor/logs/
```

### 방법 2: Git을 통해 (결과를 커밋할 경우)
```bash
# HPC에서 결과를 git에 추가
cd /scratch/x2026a01/gait_wearable_sensor
git add results/error_analysis/ results/confusion_analysis/
git commit -m "Add Phase 1 analysis results"
git push origin main

# 로컬에서 pull
cd D:/gait_wearable_sensor
git pull origin main
```

**주의**: 결과 파일이 큰 경우 (PNG ~2MB) Git에 커밋하지 말고 scp 사용 권장

---

## 🔧 문제 해결

### 1. Git pull 충돌
```bash
# 로컬 변경사항 확인
git status

# 로컬 변경사항 백업
git stash

# Pull 재시도
git pull origin main

# 백업 복원 (필요시)
git stash pop
```

### 2. 모델 파일 없음
```bash
# 모델 파일 확인
ls -la /scratch/x2026a01/gait_wearable_sensor/models/OA_Screening_best.pth

# 없으면 baseline 학습 먼저
nohup python -u src/train_baseline_hpc.py --task OA_Screening > logs/oa.log 2>&1 &
```

### 3. seaborn 모듈 없음
```bash
# pip 설치
pip install --user seaborn

# 설치 확인
python -c "import seaborn; print(seaborn.__version__)"
```

### 4. 디렉토리 권한 오류
```bash
# 결과 디렉토리 생성
mkdir -p /scratch/x2026a01/gait_wearable_sensor/results/error_analysis
mkdir -p /scratch/x2026a01/gait_wearable_sensor/results/confusion_analysis
mkdir -p /scratch/x2026a01/gait_wearable_sensor/logs

# 권한 확인
ls -ld /scratch/x2026a01/gait_wearable_sensor/results/
```

### 5. 프로세스 종료 (필요시)
```bash
# 실행 중인 프로세스 확인
ps aux | grep analyze

# PID로 종료
kill <PID>

# 또는 이름으로 종료
pkill -f analyze_errors
pkill -f analyze_confusion
```

---

## 📊 예상 실행 시간

| 단계 | 시간 |
|------|------|
| git pull | 5초 |
| Error Analysis | 5-10분 |
| Confusion Analysis | 5-10분 |
| 결과 다운로드 (scp) | 10-30초 |

**총 예상 시간**: 10-20분 (백그라운드 실행)

---

## ✅ 실행 체크리스트

### HPC 작업
- [ ] SSH 접속 완료
- [ ] `git pull origin main` 완료
- [ ] Error Analysis 실행 (`nohup ... &`)
- [ ] Confusion Analysis 실행 (`nohup ... &`)
- [ ] 로그 확인 (tail -f logs/*.log)
- [ ] 완료 확인 ("DONE" 메시지 확인)
- [ ] 결과 파일 생성 확인 (ls results/)

### 로컬 작업
- [ ] 결과 다운로드 (scp)
- [ ] PNG 파일 확인 (시각화)
- [ ] JSON 파일 확인 (수치 결과)
- [ ] 로그 파일 다운로드

---

## 🎯 결과 확인 방법

### Error Analysis 결과
1. **JSON 파일** (`OA_Screening_error_analysis.json`):
   - 전체 성능 (AUC, Balanced Accuracy)
   - False Positive 통계
   - False Negative 통계
   - 개선 방향

2. **PNG 파일** (`OA_Screening_error_analysis.png`):
   - 7개 subplot 시각화
   - Confusion Matrix, ROC, PR Curve
   - 확률 분포, 에러 비교, Box Plot

### Confusion Analysis 결과
1. **JSON 파일** (`OA_Screening_confusion_analysis.json`):
   - 7가지 최적 임계값
   - 각 임계값별 성능 지표
   - 임상 권장사항

2. **PNG 파일** (`OA_Screening_confusion_analysis.png`):
   - 10+ subplot 시각화
   - ROC with optimal points
   - Sensitivity vs Specificity
   - Youden's Index, F1 Score
   - 4가지 임계값별 Confusion Matrix

---

## 📝 다음 단계

Phase 1 완료 후:
1. **결과 해석**: JSON 파일과 시각화 분석
2. **Phase 2 준비**: 시간 패턴 분석, 질환 심각도 분석
3. **로컬 업데이트**: `git pull` 로 최신 코드 유지

**참고 문서**:
- `NEXT_ANALYSIS_PLAN.md`: 전체 분석 계획 (Phase 1-4)
- `DATASET_DOCUMENTATION.md`: 데이터셋 상세 설명
- `RUN_PHASE1_HPC.md`: 상세 실행 가이드

---

**GitHub 저장소**: https://github.com/Youngkwon-Lee/gait_analysis
**Last Updated**: 2026-01-08
