# 📊 Clinical Gait Signals Dataset Documentation

**작성일**: 2026-01-08
**데이터셋**: Clinical Gait Signals (Nature Scientific Data 2025)
**프로젝트**: Gait Wearable Sensor Analysis

---

## 🎯 데이터셋 개요

### 기본 정보
- **출처**: Nature Scientific Data 2025
- **총 규모**: 179명, 800 trials
- **측정 환경**: 평지 보행 (10m 직선, 실내, 편안한 속도)
- **센서**: 4개 IMU (Inertial Measurement Unit)
- **샘플링 속도**: 100Hz

---

## 📁 데이터 구조

### 1. 폴더 구조
```
dataset/data/
├── healthy/
│   └── HS/              # Healthy Subjects (73명, 360 trials)
├── ortho/
│   ├── HOA/             # Hip Osteoarthritis (15명, 74 trials)
│   └── KOA/             # Knee Osteoarthritis (18명, 78 trials)
└── neuro/
    ├── PD/              # Parkinson's Disease (24명, 160 trials)
    └── CVA/             # Cerebrovascular Accident (49명, 128 trials)
```

### 2. 코호트별 데이터 규모

| 코호트 | 질환명 | 환자 수 | Trial 수 | 비고 |
|--------|--------|---------|----------|------|
| **HS** | 건강한 사람 | 73명 | 360 trials | Control group |
| **HOA** | 고관절 골관절염 | 15명 | 74 trials | Hip OA |
| **KOA** | 무릎 골관절염 | 18명 | 78 trials | Knee OA |
| **PD** | 파킨슨병 | 24명 | 160 trials | Neurological |
| **CVA** | 뇌졸중 | 49명 | 128 trials | Neurological |
| **합계** | - | **179명** | **800 trials** | - |

---

## 🎯 분류 작업 (Classification Tasks)

### OA Screening (골관절염 스크리닝)
- **Class 0 (Healthy)**: HS (73명, 360 trials)
- **Class 1 (OA)**: HOA + KOA (33명, 152 trials)
- **목적**: 건강한 사람과 골관절염 환자 구분
- **클래스 비율**: 약 2.4:1 (불균형)

### PD Screening (파킨슨병 스크리닝)
- **Class 0 (Healthy)**: HS (73명, 360 trials)
- **Class 1 (PD)**: PD (24명, 160 trials)
- **목적**: 건강한 사람과 파킨슨병 환자 구분

### CVA Detection (뇌졸중 감지)
- **Class 0 (Healthy)**: HS (73명, 360 trials)
- **Class 1 (CVA)**: CVA (49명, 128 trials)
- **목적**: 건강한 사람과 뇌졸중 환자 구분

---

## 📏 데이터 형식

### 1. Window Size: 3초

**계산**:
```
Window Size = 300 samples
Sampling Rate = 100 Hz
→ 300 ÷ 100 = 3초
```

**3초를 선택한 이유**:
- ✅ 정상 보행 1-2 걸음 (gait cycle) 포함
- ✅ 충분한 패턴 캡처 가능
- ✅ 실시간 분석에 적합 (너무 길지 않음)
- ✅ 배터리 효율과 정확도 균형

### 2. Windowing Process (50% Overlap)

**Parameters**:
- **Window Size**: 300 samples (3초)
- **Stride**: 150 samples (50% overlap)
- **평균 Trial 길이**: ~1500 samples (15초)

**예시**:
```
원본 Trial: 1500 samples (15초 보행)
↓
Window 1: [0:300]       → 0.0-3.0초
Window 2: [150:450]     → 1.5-4.5초
Window 3: [300:600]     → 3.0-6.0초
Window 4: [450:750]     → 4.5-7.5초
...
Window 9: [1200:1500]   → 12.0-15.0초

→ 한 Trial에서 약 9-10개 Window 생성
```

**Overlap 이유**:
- 데이터 증강 (Data Augmentation)
- 경계 효과 완화 (Boundary Effect)
- 연속성 보존 (Temporal Continuity)

### 3. Tensor 형태

**각 Window**:
```python
shape: (4, 6, 300)
       ↓  ↓  ↓
    sensors × channels × timesteps

실제 Tensor: torch.Size([4, 6, 300])
```

**배치 형태** (학습 시):
```python
shape: (batch_size, 4, 6, 300)
예: (32, 4, 6, 300)  # Batch size 32
```

---

## 📍 센서 배치

### 4개 IMU 센서 위치

| 센서 코드 | 위치 | 영어명 | 측정 목적 |
|-----------|------|--------|-----------|
| **HE** | 머리 | Head | 상체 움직임, 자세 |
| **LB** | 허리 | Lower Back | 몸통 중심 움직임 |
| **LF** | 왼발 | Left Foot | 발 착지, 보행 패턴 |
| **RF** | 오른발 | Right Foot | 발 착지, 보행 패턴 |

**센서 중요도** (모델 기반):
1. **LF** (왼발) - AUC drop 0.0042
2. **RF** (오른발) - AUC drop 0.0026
3. **HE** (머리) - AUC drop 0.0022
4. **LB** (허리) - AUC drop 0.0006

---

## 📊 측정 채널

### 6개 채널 (각 센서당)

| 채널 | 센서 타입 | 측정 내용 | 단위 | 범위 |
|------|-----------|-----------|------|------|
| **Acc_X** | Accelerometer | 전후 가속도 | m/s² | ±16g |
| **Acc_Y** | Accelerometer | 좌우 가속도 | m/s² | ±16g |
| **Acc_Z** | Accelerometer | 상하 가속도 | m/s² | ±16g |
| **Gyr_X** | Gyroscope | 전후 회전 | °/s | ±2000°/s |
| **Gyr_Y** | Gyroscope | 좌우 회전 | °/s | ±2000°/s |
| **Gyr_Z** | Gyroscope | 상하 회전 | °/s | ±2000°/s |

**참고**: Magnetometer (자력계) 데이터는 센서 교란 방지를 위해 제외

**채널 중요도** (모델 기반):
1. **Acc_X** - AUC drop 0.0949 (가장 중요!)
2. **Acc_Z** - AUC drop 0.0736
3. **Acc_Y** - AUC drop 0.0395
4. **Gyr_X** - AUC drop 0.0177
5. **Gyr_Z** - AUC drop 0.0163
6. **Gyr_Y** - AUC drop 0.0132

---

## 🔄 Train/Test Split

### Subject-wise Split (중요!)

**데이터 유출 방지** (Data Leakage Prevention):
- ❌ Trial 단위 분할 → 같은 환자의 다른 Trial이 Train/Test에 섞임
- ✅ **Subject 단위 분할** → 환자 전체가 Train 또는 Test로만 들어감

**OA Screening 예시**:
```
전체: 106명 (HS 73명 + OA 33명)
↓ 80/20 Split (Subject-wise)
Train: ~85명 (HS ~58명 + OA ~27명)
Test: ~21명 (HS ~15명 + OA ~6명)
```

### Trial 수 (Split 후)

| 구분 | Subject 수 | Trial 수 | Window 수 (추정) |
|------|------------|----------|------------------|
| **Train** | ~85명 | ~410 trials | ~4,100 windows |
| **Test** | ~21명 | ~102 trials | ~1,020 windows |

**Window 수 계산**:
- 평균 Trial당 10개 window (50% overlap)
- Train: 410 trials × 10 ≈ 4,100 windows
- Test: 102 trials × 10 ≈ 1,020 windows

---

## 🎯 데이터 품질

### 전처리 (Preprocessing)

1. **정규화** (Normalization)
   - 각 채널별로 Z-score 정규화
   - Mean = 0, Std = 1

2. **결측치 처리**
   - 센서 오류 시 해당 Trial 제외
   - 모든 Trial은 완전한 데이터만 사용

3. **이상치 제거**
   - 물리적으로 불가능한 값 (±16g 초과 등) 제거
   - 센서 오작동 Trial 제외

### 데이터 증강 (Data Augmentation)

**현재 사용**:
- ✅ 50% Overlap Windowing → 데이터 2배 증강 효과

**미사용** (향후 고려 가능):
- ❌ Time Warping
- ❌ Magnitude Warping
- ❌ Noise Injection

---

## 📈 성능 벤치마크

### OA Screening 성능 (Baseline)

| 지표 | 값 | 설명 |
|------|-----|------|
| **AUC** | 0.9923 | 거의 완벽한 분류 |
| **Balanced Accuracy** | 0.9593 | 클래스 불균형 고려 |
| **Sensitivity** | ~0.95 | OA 환자 정확히 찾아냄 |
| **Specificity** | ~0.96 | 건강한 사람 정확히 구분 |

---

## 🔍 주요 발견사항

### 1. 센서 중요도
- **발 센서 (LF, RF)** > 머리/허리 센서
- 보행 이상은 발에서 가장 명확히 나타남

### 2. 채널 중요도
- **가속도계 (Acc)** > 자이로스코프 (Gyr)
- 특히 **Acc_X (전후 가속도)**가 가장 중요
- 직선 움직임이 회전보다 더 유용한 정보

### 3. 시간 길이
- **3초 Window**가 적절
- 더 길면 (5초+): 배터리 낭비
- 더 짧으면 (1초): 패턴 불충분

---

## 📚 데이터셋 한계

### 1. 클래스 불균형
- HS : OA = 2.4 : 1
- 해결: Balanced Accuracy, Class Weights 사용

### 2. 단일 환경
- 실내 평지만 측정
- 실제: 계단, 경사, 야외 등 다양

### 3. 단일 센서 제조사
- 센서 간 차이 (Inter-sensor Variability) 미검증
- Cross-dataset 검증 필요

### 4. 제한된 인구통계
- 나이, 성별, BMI 분포 편향 가능
- 다양성 확보 필요

---

## 🎯 권장 사항

### 연구용
1. **통계 분석**: Effect Size (Gyr_X 중요)
2. **병리학적 해석**: 회전 패턴 변화 설명
3. **논문 작성**: 두 방법 모두 제시

### 제품용
1. **모델 배포**: Permutation Importance (Acc_X 중요)
2. **센서 설계**: LF/RF 우선, Acc 중심
3. **실시간 처리**: 3초 Window, 50% Overlap

### 미래 연구
1. **다양한 환경**: 야외, 계단, 경사
2. **다른 질환**: 초기 OA, 경증 PD
3. **Cross-dataset**: 다른 센서, 다른 프로토콜
4. **Attention 분석**: 모델이 정확히 어느 부분 보는지

---

## 📖 참고 문헌

- **데이터셋 논문**: Nature Scientific Data 2025
- **모델 논문**: arXiv 2511.02047 (Multi-Stream Attention CNN)
- **Feature Importance 분석**: [results/importance_comparison/](results/importance_comparison/)

---

**Last Updated**: 2026-01-08
**다음 분석**: Phase 1 - Error Analysis & Confusion Analysis
