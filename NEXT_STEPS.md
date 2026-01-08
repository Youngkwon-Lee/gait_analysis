# Next Steps - 결과 다운로드 및 시각화

HPC 훈련이 완료되었으므로 이제 결과를 다운로드하고 시각화/분석하겠습니다.

---

## 📥 Step 1: HPC 결과 다운로드 (수동)

Windows에서 SSH 호스트 해석 문제로 **수동 다운로드**가 필요합니다.

### 방법 선택:

#### Option A: WinSCP (가장 쉬움 - 권장)
1. WinSCP 다운로드: https://winscp.net/eng/download.php
2. 접속 정보:
   ```
   호스트: VM1212121914
   포트: 22
   사용자: gun3856
   비밀번호: (본인 비밀번호)
   ```
3. 원격 경로: `/home2/gun3856/gait_analysis/results/`
4. 로컬 경로: `D:\gait_wearable_sensor\results\`
5. 다음 4개 CSV 파일 다운로드:
   - `dl_baseline_results_20260107_144801.csv` (PD Screening)
   - `dl_baseline_results_20260107_155554.csv` (OA Screening)
   - `dl_baseline_results_20260107_162124.csv` (CVA Detection)
   - `dl_baseline_results_20260107_165320.csv` (PD vs CVA)

#### Option B: HPC 터미널에서 압축 → 수동 다운로드
HPC 터미널에서:
```bash
cd ~/gait_analysis/results
tar -czf all_results.tar.gz dl_baseline_results_*.csv
```

그 다음 WinSCP로 `all_results.tar.gz` 하나만 다운로드 후:
```bash
cd D:/gait_wearable_sensor/results
tar -xzf all_results.tar.gz
```

**상세 가이드**: `HPC_DOWNLOAD_INSTRUCTIONS.md` 참조

---

## ✅ Step 2: 다운로드 검증

파일을 다운로드한 후 검증:

```bash
cd D:/gait_wearable_sensor
python src/verify_results.py
```

**예상 출력**:
```
✅ 4개 결과 파일 발견
✅ Task: PD_Screening - ROC-AUC: 0.963
✅ Task: OA_Screening - ROC-AUC: 0.908
✅ Task: CVA_Detection - ROC-AUC: 0.986
✅ Task: PD_vs_CVA - ROC-AUC: 0.934
```

---

## 📊 Step 3: 결과 시각화

검증 완료 후 시각화 실행:

```bash
python src/visualize_results.py
```

**생성되는 파일들**:
- `visualizations/roc_curves_all_tasks.png` - ROC 커브 비교 (우리 vs 논문)
- `visualizations/confusion_matrices_all_tasks.png` - 4개 Task Confusion Matrix
- `visualizations/performance_comparison.png` - AUC/Balanced Acc 막대그래프
- `visualizations/PERFORMANCE_COMPARISON_TABLE.md` - 상세 성능 비교표

---

## 🔍 Step 4: 성능 분석

시각화 완료 후:

1. **ROC 커브 분석**:
   - 각 Task별 ROC-AUC 개선도 확인
   - False Positive Rate vs True Positive Rate 트레이드오프 분석

2. **Confusion Matrix 분석**:
   - False Positive/Negative 비율 확인
   - Sensitivity (민감도) vs Specificity (특이도) 분석

3. **베이스라인 비교**:
   - 4개 Task 중 어떤 Task에서 개선/악화되었는지 확인
   - 개선도가 큰 Task의 특징 파악

4. **오답 분석** (추가 작업 필요):
   - 어떤 샘플이 misclassified 되었는지 확인
   - 오분류 패턴 찾기

---

## 📝 Step 5: 문서화

분석 완료 후:

1. **RESULTS_SUMMARY.md** 업데이트 (시각화 링크 추가)
2. **BASELINE_COMPARISON.md** 보완 (차이점 원인 분석)
3. **GitHub에 커밋**:
   ```bash
   git add visualizations/ RESULTS_SUMMARY.md BASELINE_COMPARISON.md
   git commit -m "feat: Add visualization and detailed analysis"
   git push origin main
   ```

---

## 🎯 예상 성능 (이미 HPC에서 확인된 결과)

| Task | Our AUC | Baseline AUC | Improvement |
|------|---------|--------------|-------------|
| PD Screening | 0.963 | 0.821 | **+17.3%** ✅ |
| OA Screening | 0.908 | 0.990 | **-8.3%** ⚠️ |
| CVA Detection | 0.986 | 0.950 | **+3.8%** ✅ |
| PD vs CVA | 0.934 | 0.657 | **+42.2%** 🔥 (BEST!) |

---

## ⚠️ 주의사항

1. **OA Screening 성능 하락** (-8.3%):
   - 원인 분석 필요
   - 베이스라인 논문이 HOA+KOA 사용한 반면 우리는 HOA만 사용했을 가능성
   - `BASELINE_COMPARISON.md` 참조

2. **Magnetometer 제외**:
   - 우리: 6 channels (Acc_XYZ, Gyr_XYZ)
   - 논문: 9 channels (Acc_XYZ, Gyr_XYZ, Mag_XYZ)
   - 의도적 제외 (센서 타입 confound 방지)

3. **Window Size**:
   - 우리: 300 samples (3초)
   - 논문: 500 samples (5초)
   - 실용성과 GPU 메모리 고려

---

## 📧 문의

분석 결과나 시각화에 문제가 있으면:
1. `verify_results.py` 재실행
2. CSV 파일 형식 확인
3. GitHub Issues에 리포트

---

**현재 단계**: Step 1 (HPC 결과 다운로드) 진행 중
**다음 단계**: 파일 다운로드 완료 후 `verify_results.py` 실행
