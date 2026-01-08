# HPC-Local 워크플로우 가이드

**목적**: 다음 세션에서도 HPC와 로컬 간 관계, 소통 방법, Git 워크플로우를 기억할 수 있도록 정리

---

## 📁 디렉토리 구조

### 로컬 (Windows)
```
D:\gait_wearable_sensor\          # 로컬 작업 디렉토리 (Git 연동)
├── src/                          # 소스 코드 (여기서 편집)
│   ├── train_baseline_hpc.py
│   ├── analyze_errors.py
│   └── analyze_confusion.py
├── results/                      # HPC 실행 결과 다운로드
│   ├── error_analysis/
│   └── confusion_analysis/
├── models/                       # 학습된 모델 (필요시 다운로드)
└── *.md                          # 문서들

GitHub Repository: https://github.com/Youngkwon-Lee/gait_analysis
```

### HPC (Linux)
```
/home2/gun3856/
├── gait_code/                    # GitHub 연동 디렉토리 (코드만)
│   ├── src/                      # git pull로 최신 코드 받기
│   ├── logs/                     # 실행 로그
│   ├── dataset -> ~/gait_analysis/dataset    # 심볼릭 링크
│   ├── models -> ~/gait_analysis/models      # 심볼릭 링크
│   └── results -> ~/gait_analysis/results    # 심볼릭 링크
│
└── gait_analysis/                # 데이터 디렉토리 (Git 제외)
    ├── dataset/                  # 실제 데이터셋 (800개 trial)
    │   └── data/
    │       ├── healthy/
    │       ├── ortho/
    │       └── neuro/
    ├── models/                   # 실제 학습된 모델 파일
    │   ├── OA_Screening_best.pt
    │   └── OA_Screening_best.pth -> OA_Screening_best.pt
    └── results/                  # 실제 분석 결과
        ├── error_analysis/
        └── confusion_analysis/
```

**핵심 포인트**:
- **gait_code/**: 코드만 (Git으로 관리)
- **gait_analysis/**: 데이터 + 모델 + 결과 (용량 큼, Git 제외)
- **심볼릭 링크**: gait_code에서 gait_analysis 데이터 접근

---

## ⚠️ Git으로 관리하는 것 vs 안 하는 것

### ✅ Git으로 관리 (git push/pull)
- **소스 코드** (`src/*.py`)
- **문서** (`*.md`)
- **설정 파일** (`.gitignore`, `requirements.txt`)

**이유**: 용량 작고, 버전 관리 필요

### ❌ Git으로 관리 안 함 (scp로 전송)
- **데이터셋** (`dataset/data/`) - 수십 GB, 변하지 않음
- **모델 파일** (`models/*.pth`, `*.pt`) - 100MB+ 용량
- **결과 파일** (`results/`) - 용량 크고 자주 변함
- **로그 파일** (`logs/*.log`) - 매 실행마다 변함

**이유**:
1. GitHub 파일 크기 제한 (100MB)
2. Git 저장소 크기 증가 방지
3. 불필요한 네트워크 트래픽 방지

### 올바른 워크플로우
```bash
# 로컬에서 코드 수정
git add src/analyze_errors.py
git push origin main

# HPC에서 코드 받기
git pull origin main

# HPC에서 실행 (모델/데이터는 이미 HPC에 있음)
python src/analyze_errors.py

# 결과만 scp로 다운로드
scp -r gun3856@VM1212121914:~/gait_code/results/error_analysis/* D:/gait_wearable_sensor/results/error_analysis/
```

---

## 🔄 Git 워크플로우

### 1. 로컬에서 코드 수정 → HPC 실행

```bash
# ===== 로컬 (Windows) =====
cd D:\gait_wearable_sensor

# 1. 코드 수정 (예: src/analyze_errors.py)
# (VS Code나 에디터로 편집)

# 2. Git 커밋 & 푸시
git add src/analyze_errors.py
git commit -m "fix: Model architecture alignment with baseline"
git push origin main

# ===== HPC (Linux) =====
# 3. HPC에서 최신 코드 받기
ssh gun3856@VM1212121914
cd ~/gait_code
git pull origin main

# 4. HPC에서 실행
export DATA_PATH=~/gait_code/dataset/data
export OUTPUT_PATH=~/gait_code/results/error_analysis
export MODEL_PATH=~/gait_code/models
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &

# 5. 로그 확인
tail -f logs/error_analysis.log

# ===== 로컬 (Windows) =====
# 6. 결과 다운로드
scp -r gun3856@VM1212121914:~/gait_code/results/error_analysis/* D:/gait_wearable_sensor/results/error_analysis/
```

### 2. HPC에서 실행 중 에러 발생 → 로컬 수정 → 재실행

```bash
# ===== HPC =====
# 1. 에러 발견 (로그 확인)
tail -100 ~/gait_code/logs/error_analysis.log

# ===== 로컬 =====
# 2. 로컬에서 코드 수정
# (에디터로 수정)

# 3. Git 푸시
git add .
git commit -m "fix: Data loading issue"
git push origin main

# ===== HPC =====
# 4. 최신 코드 받기
cd ~/gait_code && git pull origin main

# 5. 다시 실행
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &
```

---

## 🚀 Phase 1 실행 가이드 (Error & Confusion Analysis)

### Quick Start (복사해서 사용)

```bash
# ===== HPC 접속 =====
ssh gun3856@VM1212121914
cd ~/gait_code

# ===== 1. 최신 코드 받기 =====
git pull origin main

# ===== 2. Error Analysis 실행 =====
export DATA_PATH=~/gait_code/dataset/data
export OUTPUT_PATH=~/gait_code/results/error_analysis
export MODEL_PATH=~/gait_code/models
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &

# PID 확인 (나중에 kill 필요시)
echo $! > logs/error_analysis.pid

# 로그 확인 (실시간)
tail -f logs/error_analysis.log

# 완료 확인 (Ctrl+C 후)
tail -50 logs/error_analysis.log

# ===== 3. Confusion Analysis 실행 =====
export DATA_PATH=~/gait_code/dataset/data
export OUTPUT_PATH=~/gait_code/results/confusion_analysis
export MODEL_PATH=~/gait_code/models
nohup python -u src/analyze_confusion.py > logs/confusion_analysis.log 2>&1 &

echo $! > logs/confusion_analysis.pid
tail -f logs/confusion_analysis.log

# ===== 4. 결과 확인 =====
ls -lh ~/gait_code/results/error_analysis/
ls -lh ~/gait_code/results/confusion_analysis/
```

### 로컬에서 결과 다운로드

```bash
# ===== Windows PowerShell =====
# Error Analysis 결과
scp -r gun3856@VM1212121914:~/gait_code/results/error_analysis/* D:/gait_wearable_sensor/results/error_analysis/

# Confusion Analysis 결과
scp -r gun3856@VM1212121914:~/gait_code/results/confusion_analysis/* D:/gait_wearable_sensor/results/confusion_analysis/

# 로그 파일도 다운로드 (필요시)
scp gun3856@VM1212121914:~/gait_code/logs/error_analysis.log D:/gait_wearable_sensor/logs/
scp gun3856@VM1212121914:~/gait_code/logs/confusion_analysis.log D:/gait_wearable_sensor/logs/
```

---

## 🔧 심볼릭 링크 설정 (최초 1회)

HPC에서 gait_code와 gait_analysis를 연결 (이미 완료했지만 참고용)

```bash
cd ~/gait_code

# 데이터셋 링크
ln -s ~/gait_analysis/dataset dataset

# 모델 링크
ln -s ~/gait_analysis/models models

# 결과 링크
ln -s ~/gait_analysis/results results

# 확인
ls -la
# dataset -> /home2/gun3856/gait_analysis/dataset
# models -> /home2/gun3856/gait_analysis/models
# results -> /home2/gun3856/gait_analysis/results
```

---

## 🐛 트러블슈팅

### 1. 모델 파일 없음
```bash
# 에러: Model not found: ~/gait_code/models/OA_Screening_best.pth
# 확인
ls -lh ~/gait_code/models/

# 해결: 모델 학습 먼저 실행
cd ~/gait_code
nohup python -u src/train_baseline_hpc.py --task OA_Screening > logs/train_oa.log 2>&1 &

# 30-60분 후 완료 확인
ls -lh ~/gait_code/models/OA_Screening_best.pt
```

### 2. GPU 메모리 부족
```bash
# 에러: RuntimeError: CUDA error: out of memory
# 해결: 코드에서 CPU 사용하도록 이미 설정됨
# Config.DEVICE = torch.device('cpu')
```

### 3. 데이터 로딩 실패
```bash
# 에러: Loaded Class 0 (HS): 0 trials
# 확인: 데이터 경로 확인
echo $DATA_PATH
ls -lh $DATA_PATH/healthy/HS/

# 해결: 환경변수 다시 설정
export DATA_PATH=~/gait_code/dataset/data
```

### 4. Git Pull 충돌
```bash
# 에러: error: Your local changes would be overwritten by merge
# 해결: HPC에서는 코드 직접 수정 안함, 로컬만 수정
git stash  # 임시 저장
git pull origin main
git stash pop  # 복원 (충돌나면 해결)
```

---

## 📊 실행 상태 모니터링

### 실행 중인 프로세스 확인
```bash
# 현재 실행 중인 Python 프로세스
ps aux | grep python

# 특정 스크립트 실행 여부
ps aux | grep analyze_errors.py

# PID 파일로 확인
cat logs/error_analysis.pid
ps -p $(cat logs/error_analysis.pid)
```

### 로그 실시간 모니터링
```bash
# 실시간 로그 (Ctrl+C로 종료)
tail -f logs/error_analysis.log

# 마지막 100줄
tail -100 logs/error_analysis.log

# 에러만 검색
grep -i error logs/error_analysis.log
grep -i "exception\|error\|fail" logs/error_analysis.log
```

### 프로세스 종료 (필요시)
```bash
# PID로 종료
kill $(cat logs/error_analysis.pid)

# 강제 종료 (응답 없을 때)
kill -9 $(cat logs/error_analysis.pid)

# 또는 프로세스 ID 직접 사용
kill 1234567
```

---

## 📝 데이터 구조 참고

### 데이터셋 구조
```
dataset/data/
├── healthy/HS/
│   ├── sub-001/
│   │   ├── gait-01/
│   │   │   ├── gait-01_meta.json      # 메타데이터
│   │   │   ├── gait-01_HE.txt         # 센서 데이터
│   │   │   ├── gait-01_LB.txt
│   │   │   ├── gait-01_LF.txt
│   │   │   └── gait-01_RF.txt
│   │   └── gait-02/
│   └── sub-002/
├── ortho/HOA/
├── ortho/KOA/
├── neuro/PD/
└── neuro/CVA/
```

### Task 정의
```python
TASKS = {
    'OA_Screening': {
        'class0': ('HS', 'healthy'),           # 건강한 사람
        'class1': [('HOA', 'ortho'), ('KOA', 'ortho')]  # 골관절염 (HOA + KOA)
    }
}
```

---

## ✅ 체크리스트

### 새 세션 시작할 때
- [ ] HPC 접속 확인: `ssh gun3856@VM1212121914`
- [ ] 디렉토리 이동: `cd ~/gait_code`
- [ ] 최신 코드 받기: `git pull origin main`
- [ ] 환경변수 설정: `export DATA_PATH=...`

### 코드 수정 후
- [ ] 로컬에서 Git commit & push
- [ ] HPC에서 git pull
- [ ] 환경변수 재설정 (필요시)
- [ ] nohup으로 백그라운드 실행
- [ ] 로그 확인으로 정상 동작 확인

### 분석 완료 후
- [ ] 로그 마지막 부분 확인 (에러 없는지)
- [ ] 결과 파일 생성 확인: `ls -lh results/`
- [ ] scp로 로컬에 다운로드
- [ ] 로컬에서 결과 검토

---

## 🔑 핵심 요약

| 작업 | 위치 | 도구 |
|------|------|------|
| **코드 수정** | 로컬 (Windows) | VS Code + Git |
| **코드 동기화** | GitHub | git push/pull |
| **모델 학습** | HPC (Linux) | train_baseline_hpc.py |
| **분석 실행** | HPC (Linux) | analyze_*.py |
| **결과 다운로드** | 로컬 ← HPC | scp |
| **결과 검토** | 로컬 (Windows) | Excel, Python |

**핵심 원칙**:
1. **로컬**: 코드 수정 + Git 관리
2. **GitHub**: 코드 중앙 저장소
3. **HPC**: 계산 + 실행 + 결과 생성
4. **로컬 ← HPC**: 결과만 다운로드 (scp)

---

## 📄 .gitignore 설정

`.gitignore` 파일에 다음을 추가하여 대용량 파일 제외:

```gitignore
# 데이터셋 (용량 큼)
dataset/
data/

# 모델 파일 (100MB+)
*.pth
*.pt
*.ckpt
*.h5
*.pkl
*.joblib

# 결과 파일
results/
outputs/
figures/
plots/

# 로그 파일
logs/
*.log
nohup.out

# Python 캐시
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# 환경
.env
.venv/
venv/
```

**확인 방법**:
```bash
# Git 추적 파일 확인
git ls-files

# 추적되지 않는 대용량 파일 확인
git status --ignored
```

---

## 📞 빠른 참조

### HPC 접속
```bash
ssh gun3856@VM1212121914
```

### Git 푸시 (로컬)
```bash
git add . && git commit -m "메시지" && git push origin main
```

### Git 풀 + 실행 (HPC)
```bash
cd ~/gait_code && git pull origin main && \
export DATA_PATH=~/gait_code/dataset/data && \
export OUTPUT_PATH=~/gait_code/results/error_analysis && \
export MODEL_PATH=~/gait_code/models && \
nohup python -u src/analyze_errors.py > logs/error_analysis.log 2>&1 &
```

### 결과 다운로드 (로컬)
```bash
scp -r gun3856@VM1212121914:~/gait_code/results/error_analysis/* D:/gait_wearable_sensor/results/error_analysis/
```

---

*마지막 업데이트: 2026-01-08*
*다음 세션에서 이 파일을 먼저 읽고 시작하세요!*
