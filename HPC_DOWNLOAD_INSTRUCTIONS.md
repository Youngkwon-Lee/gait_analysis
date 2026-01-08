# HPC 결과 다운로드 가이드

Windows에서 SSH 연결 문제가 있을 수 있으므로 수동 다운로드 방법을 제공합니다.

## Option 1: WinSCP 사용 (권장)

### 1단계: WinSCP 다운로드
- https://winscp.net/eng/download.php
- 설치 후 실행

### 2단계: HPC 접속 설정
```
호스트 이름: VM1212121914
포트 번호: 22
사용자 이름: gun3856
비밀번호: (사용자 비밀번호)
```

### 3단계: 파일 다운로드
원격 경로 (HPC):
```
/home2/gun3856/gait_analysis/results/
```

로컬 경로 (Windows):
```
D:\gait_wearable_sensor\results\
```

다운로드 대상 파일:
- `dl_baseline_results_20260107_144801.csv` (PD Screening)
- `dl_baseline_results_20260107_155554.csv` (OA Screening)
- `dl_baseline_results_20260107_162124.csv` (CVA Detection)
- `dl_baseline_results_20260107_165320.csv` (PD vs CVA)

---

## Option 2: Git Bash 사용

### 1단계: HPC에서 압축
HPC 터미널에서 실행:
```bash
cd ~/gait_analysis/results
tar -czf all_results.tar.gz dl_baseline_results_*.csv
ls -lh all_results.tar.gz
```

### 2단계: 로컬에서 다운로드
Windows Git Bash에서 실행:
```bash
cd /d/gait_wearable_sensor/results
scp gun3856@VM1212121914:~/gait_analysis/results/all_results.tar.gz ./
```

### 3단계: 압축 해제
```bash
tar -xzf all_results.tar.gz
ls -lh dl_baseline_results_*.csv
```

---

## Option 3: HPC에서 GitHub 경유

### 1단계: HPC에서 결과를 GitHub에 업로드
HPC 터미널에서:
```bash
cd ~/gait_code
git add results/dl_baseline_results_*.csv
git commit -m "Add training results (CSV only)"
git push origin main
```

### 2단계: 로컬에서 pull
Windows에서:
```bash
cd D:/gait_wearable_sensor
git pull origin main
```

**주의**: CSV 파일만 커밋하세요. 모델 파일(.pt)은 너무 커서 제외됩니다.

---

## 다운로드 완료 확인

다음 명령어로 파일이 있는지 확인:
```bash
ls -lh D:/gait_wearable_sensor/results/dl_baseline_results_*.csv
```

예상 출력:
```
-rw-r--r-- 1 YK 197121 XXXK  1월  7 14:48 dl_baseline_results_20260107_144801.csv
-rw-r--r-- 1 YK 197121 XXXK  1월  7 15:55 dl_baseline_results_20260107_155554.csv
-rw-r--r-- 1 YK 197121 XXXK  1월  7 16:21 dl_baseline_results_20260107_162124.csv
-rw-r--r-- 1 YK 197121 XXXK  1월  7 16:53 dl_baseline_results_20260107_165320.csv
```

---

## 다음 단계: 시각화

결과 파일 다운로드 완료 후:
```bash
cd D:/gait_wearable_sensor
python src/visualize_results.py
```

생성될 파일:
- `visualizations/roc_curves_all_tasks.png` - ROC 커브 비교
- `visualizations/confusion_matrices_all_tasks.png` - Confusion Matrix
- `visualizations/performance_comparison.png` - 성능 비교 막대그래프
- `visualizations/PERFORMANCE_COMPARISON_TABLE.md` - 상세 성능 비교표
