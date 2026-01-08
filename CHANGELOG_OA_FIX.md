# OA Screening Task 수정 - HOA+KOA 통합

## 변경 이유

베이스라인 논문과의 비교에서 **OA Screening 성능이 -8.3% 하락**한 이유가 Task 정의 차이 때문임을 확인:

- **베이스라인 논문**: HS vs (HOA + KOA)
- **이전 우리 구현**: HS vs HOA only
- **수정 후**: HS vs (HOA + KOA) ← 베이스라인과 동일

---

## 코드 변경사항

### 1. Task 정의 수정 (Line 70)

**Before**:
```python
'OA_Screening': {'class0': ('HS', 'healthy'), 'class1': ('HOA', 'ortho')}
```

**After**:
```python
'OA_Screening': {'class0': ('HS', 'healthy'), 'class1': [('HOA', 'ortho'), ('KOA', 'ortho')]}
```

---

### 2. 데이터 로더 수정 (Line 306-344)

**기능 추가**:
- `class0`, `class1`이 리스트일 경우 여러 cohort 처리
- 여러 cohort의 trial을 하나의 class로 통합
- Backward compatible (기존 단일 cohort도 지원)

**처리 로직**:
```python
if isinstance(class1_config, list):
    # Multiple cohorts (e.g., HOA + KOA)
    trials1 = []
    for cohort, group in class1_config:
        cohort_trials = get_trial_paths(cohort, group)
        trials1.extend(cohort_trials)
else:
    # Single cohort (e.g., PD, CVA)
    cohort1, group1 = class1_config
    trials1 = get_trial_paths(cohort1, group1)
```

---

## 예상 효과

### 데이터 증가
- **HOA only**: ~100 trials
- **HOA + KOA**: ~200 trials (약 2배)
- 더 많은 데이터로 학습 → 성능 향상 예상

### 성능 예측
- **현재 (HOA only)**: AUC 0.908
- **목표 (HOA+KOA)**: AUC 0.950+ (베이스라인 0.990에 근접)

---

## 실행 방법

### HPC에서 재실험

```bash
cd ~/gait_code

# 1. 수정된 코드 pull
git pull origin main

# 2. OA Screening만 재실험
nohup python -u src/train_baseline_hpc.py --task OA_Screening > logs/oa_fixed.log 2>&1 &

# 3. 진행 상황 확인
tail -f logs/oa_fixed.log
```

---

## 비교 분석 예정

실험 완료 후 다음 비교 수행:

| 버전 | Task 정의 | 예상 AUC | 베이스라인 대비 |
|------|-----------|----------|----------------|
| **V1 (이전)** | HS vs HOA | 0.908 | -8.3% ❌ |
| **V2 (수정)** | HS vs (HOA+KOA) | 0.950+ | ~-4% 예상 |
| **Baseline** | HS vs (HOA+KOA) | 0.990 | 0% |

---

## 기타 Task는 그대로 유지

- ✅ PD_Screening: HS vs PD (단일 cohort)
- ✅ CVA_Detection: HS vs CVA (단일 cohort)
- ✅ PD_vs_CVA: PD vs CVA (단일 cohort)

---

**수정일**: 2026-01-08
**수정자**: YK (with Claude Code)
**상태**: 코드 수정 완료, HPC 재실험 대기 중
