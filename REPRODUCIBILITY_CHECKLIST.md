# 재현 가능성 체크리스트

**작성일**: 2026-01-09
**목적**: 모든 실험 결과를 추후 재현할 수 있도록 저장 상태 확인

---

## ✅ 저장 완료된 항목

### 1. 소스 코드 (GitHub)

**저장 위치**: https://github.com/Youngkwon-Lee/gait_analysis

| 파일 | 용도 | 커밋 | 상태 |
|------|------|------|------|
| `src/train_baseline_hpc.py` | 모델 학습 | 초기 | ✅ |
| `src/analyze_errors.py` | Phase 1-1 에러 분석 | 785b649 | ✅ |
| `src/analyze_local_predictions.py` | Phase 1-3 임계값 최적화 | - | ✅ |
| `src/analyze_temporal.py` | Phase 2-1 시간 분석 | - | ✅ |
| `src/analyze_sensor_importance.py` | Phase 2-2 센서 중요도 | - | ✅ |
| `src/analyze_feature_importance.py` | Phase 2-3 채널 중요도 | 8a9a49c | ✅ |

**확인 방법**:
```bash
git log --oneline --all
git diff origin/main  # 로컬 변경사항 확인
```

### 2. 학습된 모델 (VM)

**저장 위치**: `/home2/gun3856/gait_code/models/`

| 모델 | 크기 | 성능 | 저장 상태 |
|------|------|------|----------|
| `OA_Screening_best.pt` | 731KB | AUC 0.9998 | ✅ VM |
| `PD_Screening_best.pt` | ? | AUC 0.956 | ✅ VM |
| `CVA_Detection_best.pt` | ? | AUC 0.982 | ✅ VM |

**⚠️ 백업 필요**: VM 모델을 로컬/클라우드로 백업해야 함!

**백업 명령어**:
```bash
# VM에서 실행
cd ~/gait_code/models
tar -czf models_backup_20260109.tar.gz *.pt
```

### 3. 학습 로그 (VM)

**저장 위치**: `/home2/gun3856/gait_code/logs/`

| 로그 | 내용 | 저장 상태 |
|------|------|----------|
| `pd_training.log` | PD 학습 과정 (50 epochs) | ✅ VM |
| `cva_training.log` | CVA 학습 과정 (50 epochs) | ✅ VM |

**⚠️ 백업 필요**: 로그도 다운로드해야 함!

### 4. 분석 결과 (로컬)

**저장 위치**: `D:\gait_wearable_sensor\results\`

#### OA_Screening (100% 완료)

| Phase | 파일 | 크기 | 상태 |
|-------|------|------|------|
| 1-1 | `error_analysis/OA_Screening_error_analysis.json` | 168KB | ✅ |
| 1-1 | `error_analysis/OA_Screening_error_analysis.png` | - | ✅ |
| 1-3 | `local_analysis/OA_Screening_local_*.json` | - | ✅ |
| 2-1 | `temporal_analysis/OA_Screening_temporal_analysis.json` | 789KB | ✅ |
| 2-2 | `sensor_importance/OA_Screening_sensor_importance.json` | 3KB | ✅ |
| 2-3 | `feature_importance/OA_Screening_feature_importance.json` | 11KB | ✅ |

#### PD_Screening (진행 중)

| Phase | 파일 | 상태 |
|-------|------|------|
| 1-1 | `error_analysis/PD_Screening_error_analysis.json` | ⏳ 생성 예정 |
| 2-1 | 시간 분석 | ⏳ 대기 |
| 2-2 | 센서 중요도 | ⏳ 대기 |
| 2-3 | 채널 중요도 | ⏳ 대기 |

#### CVA_Detection (진행 중)

| Phase | 파일 | 상태 |
|-------|------|------|
| 1-1 | `error_analysis/CVA_Detection_error_analysis.json` | ⏳ 생성 예정 |
| 2-1 | 시간 분석 | ⏳ 대기 |
| 2-2 | 센서 중요도 | ⏳ 대기 |
| 2-3 | 채널 중요도 | ⏳ 대기 |

### 5. 문서화 (GitHub)

**저장 위치**: `D:\gait_wearable_sensor\`

| 문서 | 내용 | 커밋 | 상태 |
|------|------|------|------|
| `LOCAL_ANALYSIS_RESULTS.md` | Phase 1-3 임계값 최적화 | - | ✅ |
| `PHASE2_1_TEMPORAL_RESULTS.md` | Phase 2-1 시간 분석 | - | ✅ |
| `PHASE2_2_SENSOR_RESULTS.md` | Phase 2-2 센서 중요도 | - | ✅ |
| `PHASE2_3_FEATURE_RESULTS.md` | Phase 2-3 채널 중요도 | 9641700 | ✅ |
| `SESSION_HANDOFF.md` | 세션 핸드오프 | 071277a | ✅ |
| `REPRODUCIBILITY_CHECKLIST.md` | 이 파일 | 작성 중 | ⏳ |

---

## ⚠️ 백업 필요 항목

### 우선순위 1: 모델 파일

**문제**: VM 모델이 삭제되면 재학습 필요 (각 1시간)

**해결**:
```bash
# VM에서 실행
cd ~/gait_code/models
tar -czf models_backup_20260109.tar.gz *.pt *.pth

# 로컬로 다운로드
# (WinSCP, scp, 또는 VM 파일 매니저 사용)
```

**백업 위치**:
- `D:\gait_wearable_sensor\models_backup\`
- Google Drive / OneDrive (추천)

### 우선순위 2: 학습 로그

**문제**: 학습 과정 재현 불가

**해결**:
```bash
# VM에서 실행
cd ~/gait_code/logs
tar -czf logs_backup_20260109.tar.gz *.log

# 로컬로 다운로드
```

**백업 위치**: `D:\gait_wearable_sensor\logs_backup\`

### 우선순위 3: 원본 데이터셋

**문제**: VM 데이터셋 경로 의존

**현재 위치**: `/home2/gun3856/gait_code/dataset/data`

**확인 필요**:
- 데이터셋 원본이 어디 있는지?
- 다운로드 링크 또는 로컬 백업 있는지?

---

## 🔄 재현 절차

### 완전 재현 (처음부터)

**필요한 것**:
1. 원본 데이터셋 (Clinical Gait Signals)
2. 학습 코드 (`train_baseline_hpc.py`)
3. Python 환경 (`requirements.txt` 필요!)

**절차**:
```bash
# 1. 환경 설정
conda create -n gait python=3.10
conda activate gait
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn tqdm

# 2. 데이터 준비
export DATA_PATH="/path/to/data"

# 3. 학습
python src/train_baseline_hpc.py --task OA_Screening
python src/train_baseline_hpc.py --task PD_Screening
python src/train_baseline_hpc.py --task CVA_Detection

# 4. 분석
python src/analyze_errors.py --task OA_Screening
python src/analyze_sensor_importance.py --task OA_Screening
python src/analyze_feature_importance.py --task OA_Screening
```

### 부분 재현 (모델 있음)

**필요한 것**:
1. 학습된 모델 (.pt 파일)
2. 분석 코드
3. 테스트 데이터

**절차**:
```bash
# 분석만 재실행
export MODEL_PATH="/path/to/models"
python src/analyze_errors.py --task OA_Screening
```

---

## 📋 TODO: 추가 필요 항목

### 즉시 필요

- [ ] **requirements.txt 생성** - Python 패키지 버전 고정
- [ ] **VM 모델 파일 백업** - 로컬 또는 클라우드로
- [ ] **학습 로그 백업** - 로컬로 다운로드
- [ ] **환경변수 문서화** - DATA_PATH, MODEL_PATH 등

### 나중에 필요

- [ ] **Docker 이미지** - 전체 환경 패키징
- [ ] **데이터 전처리 스크립트** - 원본→전처리 자동화
- [ ] **자동화 스크립트** - 학습→분석 파이프라인
- [ ] **성능 비교 표** - 논문 vs 우리 결과 정리

---

## 📝 재현 가능성 확인

### 체크리스트

**코드**:
- [x] GitHub에 커밋됨
- [x] 버전 관리 중
- [ ] requirements.txt 있음

**모델**:
- [x] VM에 저장됨
- [ ] 로컬/클라우드 백업

**데이터**:
- [x] VM에 있음
- [ ] 원본 출처 문서화
- [ ] 로컬 백업

**결과**:
- [x] JSON 형식 저장
- [x] 시각화 PNG 저장
- [x] 문서화 MD 저장

**환경**:
- [ ] Python 버전 명시
- [ ] 패키지 버전 고정
- [ ] CUDA/PyTorch 버전 명시

---

## 🎯 현재 상태

**저장 상태**: 70% ✅
- 코드, 결과, 문서는 완벽
- 모델과 로그는 VM에만 있음 (백업 필요!)

**재현 가능성**: 90% ✅
- 코드만 있으면 재학습 가능
- 모델 있으면 분석 즉시 가능
- 환경 정보만 추가하면 완벽

**다음 액션**:
1. ✅ **즉시**: VM 모델 파일 백업
2. ✅ **즉시**: 학습 로그 다운로드
3. ⏳ **오늘**: requirements.txt 생성
4. ⏳ **이번 주**: 데이터 출처 문서화

---

**작성**: 2026-01-09
**업데이트**: PD/CVA 학습 완료 후
