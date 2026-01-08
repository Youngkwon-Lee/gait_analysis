"""
HPC 결과 파일 검증 스크립트

다운로드한 CSV 파일이 올바른 형식인지 확인하고 간단한 요약 제공
"""

import pandas as pd
from pathlib import Path
import json

# 경로 설정
RESULTS_DIR = Path("D:/gait_wearable_sensor/results")

# 예상되는 태스크
EXPECTED_TASKS = ['PD_Screening', 'OA_Screening', 'CVA_Detection', 'PD_vs_CVA']


def verify_results():
    """결과 파일 검증 및 요약"""

    print("=" * 80)
    print("HPC 훈련 결과 검증 스크립트")
    print("=" * 80)

    # CSV 파일 찾기
    csv_files = sorted(RESULTS_DIR.glob("dl_baseline_results_*.csv"))

    if not csv_files:
        print("\n❌ 결과 파일이 없습니다!")
        print(f"경로: {RESULTS_DIR}")
        print("\n다음 방법 중 하나로 다운로드하세요:")
        print("1. HPC_DOWNLOAD_INSTRUCTIONS.md 참조")
        print("2. WinSCP 사용")
        print("3. Git Bash에서 scp 사용")
        return False

    print(f"\n✅ {len(csv_files)}개 결과 파일 발견:")

    # 각 파일 검증
    found_tasks = {}

    for csv_file in csv_files:
        print(f"\n📄 {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)

            # 필수 컬럼 확인
            required_cols = ['task', 'roc_auc', 'balanced_accuracy']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"  ⚠️  누락된 컬럼: {missing_cols}")
                continue

            # 태스크 정보 추출
            if len(df) > 0:
                task_name = df['task'].iloc[0]
                found_tasks[task_name] = {
                    'file': csv_file.name,
                    'auc': df['roc_auc'].iloc[0],
                    'balanced_acc': df['balanced_accuracy'].iloc[0],
                    'samples': len(df)
                }

                print(f"  ✅ Task: {task_name}")
                print(f"  ✅ ROC-AUC: {df['roc_auc'].iloc[0]:.3f}")
                print(f"  ✅ Balanced Accuracy: {df['balanced_accuracy'].iloc[0]:.3f}")
                print(f"  ✅ Samples: {len(df)}")

                # 추가 컬럼 확인
                if 'sensitivity' in df.columns:
                    print(f"  ✅ Sensitivity: {df['sensitivity'].iloc[0]:.3f}")
                if 'specificity' in df.columns:
                    print(f"  ✅ Specificity: {df['specificity'].iloc[0]:.3f}")

        except Exception as e:
            print(f"  ❌ 오류: {e}")

    # 요약
    print("\n" + "=" * 80)
    print("검증 요약")
    print("=" * 80)

    missing_tasks = [task for task in EXPECTED_TASKS if task not in found_tasks]

    if missing_tasks:
        print(f"\n⚠️  누락된 태스크: {missing_tasks}")
    else:
        print("\n✅ 모든 4개 태스크 결과 확인됨!")

    # 성능 요약 테이블
    if found_tasks:
        print("\n성능 요약:")
        print(f"{'Task':<20} {'AUC':<10} {'Balanced Acc':<15} {'파일명'}")
        print("-" * 80)

        for task_name in EXPECTED_TASKS:
            if task_name in found_tasks:
                info = found_tasks[task_name]
                print(f"{task_name:<20} {info['auc']:<10.3f} {info['balanced_acc']:<15.3f} {info['file']}")

    print("\n" + "=" * 80)

    if len(found_tasks) == 4:
        print("✅ 검증 완료! 시각화를 진행하세요:")
        print("   python src/visualize_results.py")
        return True
    else:
        print("⚠️  일부 결과가 누락되었습니다.")
        print("   모든 결과를 다운로드한 후 다시 실행하세요.")
        return False


if __name__ == "__main__":
    success = verify_results()
    exit(0 if success else 1)
