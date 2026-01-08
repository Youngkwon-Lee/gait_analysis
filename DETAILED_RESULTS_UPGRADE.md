# 세부 결과 저장 기능 추가

## 문제점

기존에는 **요약 성능 지표만** 저장되어 세부 분석이 불가능했습니다:

**저장되던 것**:
- ✅ ROC-AUC (scalar)
- ✅ Balanced Accuracy (scalar)
- ✅ Sensitivity (scalar)
- ✅ Specificity (scalar)

**저장되지 않았던 것** (시각화/분석 필수):
- ❌ ROC Curve 데이터 (FPR, TPR 포인트)
- ❌ Confusion Matrix (tn, fp, fn, tp)
- ❌ 샘플별 Predictions
- ❌ 샘플별 True Labels

---

## 해결 방법

### 1. evaluate() 함수 수정

**추가된 반환값**:
```python
return {
    # 기존 (요약 지표)
    'roc_auc': roc_auc,
    'balanced_acc': bal_acc,
    'sensitivity': sensitivity,
    'specificity': specificity,

    # 신규 (Confusion Matrix)
    'tn': int(tn),
    'fp': int(fp),
    'fn': int(fn),
    'tp': int(tp),

    # 신규 (ROC Curve)
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'thresholds': thresholds.tolist(),

    # 신규 (Raw Data)
    'predictions': all_preds.tolist(),
    'true_labels': all_labels.tolist()
}
```

---

### 2. 이중 저장 방식

#### CSV 파일 (요약 지표만)
```
dl_baseline_results_YYYYMMDD_HHMMSS.csv
```
- Task name
- ROC-AUC, Balanced Accuracy, Sensitivity, Specificity
- Confusion Matrix (tn, fp, fn, tp)
- **용도**: 빠른 성능 비교, 스프레드시트 분석

#### JSON 파일 (세부 데이터 포함)
```
dl_baseline_detailed_YYYYMMDD_HHMMSS.json
```
- 위 CSV 내용 전부
- ROC Curve 전체 데이터 (FPR, TPR, Thresholds)
- 모든 샘플의 Predictions
- 모든 샘플의 True Labels
- **용도**: ROC 커브 그리기, 오답 분석, 상세 시각화

---

### 3. .gitignore 수정

**Before**:
```
results/*.json  # JSON도 무시됨
```

**After**:
```
# results/*.json  # JSON 허용 (세부 분석 필요)
```

---

## 사용 예시

### CSV로 빠른 비교
```python
import pandas as pd

df = pd.read_csv('dl_baseline_results_20260108_123456.csv')
print(df[['task', 'roc_auc', 'balanced_acc']])
```

### JSON으로 ROC Curve 그리기
```python
import json
import matplotlib.pyplot as plt

with open('dl_baseline_detailed_20260108_123456.json', 'r') as f:
    results = json.load(f)

for task_result in results:
    plt.plot(task_result['fpr'], task_result['tpr'],
             label=f"{task_result['task']} (AUC={task_result['roc_auc']:.3f})")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
```

### Confusion Matrix 시각화
```python
import seaborn as sns
import matplotlib.pyplot as plt

for task_result in results:
    cm = [[task_result['tn'], task_result['fp']],
          [task_result['fn'], task_result['tp']]]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{task_result['task']} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```

### 오답 분석
```python
# 틀린 샘플 찾기
predictions = task_result['predictions']
true_labels = task_result['true_labels']

errors = [(i, pred, true) for i, (pred, true) in enumerate(zip(predictions, true_labels))
          if (pred > 0.5) != true]

print(f"Total errors: {len(errors)}")
print("First 5 errors:")
for idx, pred, true in errors[:5]:
    print(f"  Sample {idx}: Predicted {pred:.3f}, True label {true}")
```

---

## 파일 크기 비교

### CSV (요약만)
- ~1 KB per task
- 4 tasks = ~4 KB

### JSON (세부 포함)
- ~500 KB - 2 MB per task (샘플 수에 따라)
- 4 tasks = ~2-8 MB
- **Git에 포함 가능** (모델 파일보다 훨씬 작음)

---

## 시각화 스크립트 업데이트 필요

기존 `visualize_results.py`와 `visualize_simple.py`를 업데이트해서 JSON 파일을 읽도록 수정:

```python
# NEW: Load from JSON
with open('dl_baseline_detailed_*.json', 'r') as f:
    detailed_results = json.load(f)

# ROC curves from actual data
for result in detailed_results:
    plt.plot(result['fpr'], result['tpr'],
             label=f"{result['task']}")
```

---

## 혜택

1. ✅ **ROC Curve 정확한 시각화** (추정 아닌 실제 데이터)
2. ✅ **Confusion Matrix 시각화** (TP/TN/FP/FN)
3. ✅ **오답 분석 가능** (어떤 샘플이 틀렸는지)
4. ✅ **Threshold 최적화** (ROC curve에서 최적 threshold 찾기)
5. ✅ **재현 가능성** (모든 데이터 저장)

---

**수정일**: 2026-01-08
**적용 버전**: OA_Screening 재실험부터
**Backward Compatible**: 기존 코드도 동작 (JSON만 추가)
