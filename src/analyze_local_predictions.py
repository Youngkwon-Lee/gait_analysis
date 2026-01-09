"""
Local Analysis using detailed_predictions
Phase 1-3: Deep dive into error patterns without HPC

Analyzes:
1. FN/FP error characteristics
2. Probability distributions
3. Additional threshold scenarios
4. Window-level patterns
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef

# Paths
BASE_PATH = Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / 'results' / 'error_analysis'
OUTPUT_PATH = BASE_PATH / 'results' / 'local_analysis'
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# Load data
print("="*80)
print("Local Prediction Analysis - Phase 1-3")
print("="*80)

with open(RESULTS_PATH / 'OA_Screening_error_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

predictions = data['detailed_predictions']
print(f"\nLoaded {len(predictions)} window predictions")

# Convert to DataFrame
df = pd.DataFrame(predictions)
print(f"\nDataFrame shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Basic statistics
print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)

print(f"\nTotal windows: {len(df)}")
print(f"OA windows (label=1): {(df['label'] == 1).sum()}")
print(f"Healthy windows (label=0): {(df['label'] == 0).sum()}")

print(f"\nCorrect predictions: {df['correct'].sum()} ({df['correct'].sum()/len(df)*100:.2f}%)")
print(f"Incorrect predictions: {(~df['correct']).sum()} ({(~df['correct']).sum()/len(df)*100:.2f}%)")

# Error type distribution
error_counts = df['error_type'].value_counts()
print(f"\nError type distribution:")
for error_type, count in error_counts.items():
    print(f"  {error_type}: {count} ({count/len(df)*100:.2f}%)")

# ============================================================================
# 1. FN/FP Error Analysis
# ============================================================================
print("\n" + "="*80)
print("1. FN/FP ERROR CHARACTERISTICS")
print("="*80)

fn_df = df[df['error_type'] == 'FN']
fp_df = df[df['error_type'] == 'FP']

print(f"\nFalse Negatives (OA → Healthy 오판):")
print(f"  Count: {len(fn_df)}")
print(f"  Mean probability: {fn_df['probability'].mean():.4f}")
print(f"  Std probability: {fn_df['probability'].std():.4f}")
print(f"  Probability range: [{fn_df['probability'].min():.4f}, {fn_df['probability'].max():.4f}]")
print(f"  Window IDs: {sorted(fn_df['window_id'].tolist())}")

print(f"\nFalse Positives (Healthy → OA 오판):")
print(f"  Count: {len(fp_df)}")
print(f"  Mean probability: {fp_df['probability'].mean():.4f}")
print(f"  Std probability: {fp_df['probability'].std():.4f}")
print(f"  Probability range: [{fp_df['probability'].min():.4f}, {fp_df['probability'].max():.4f}]")
print(f"  Window IDs: {sorted(fp_df['window_id'].tolist())}")

# ============================================================================
# 2. Probability Distribution Analysis
# ============================================================================
print("\n" + "="*80)
print("2. PROBABILITY DISTRIBUTION")
print("="*80)

tp_df = df[df['error_type'] == 'TP']
tn_df = df[df['error_type'] == 'TN']

print(f"\nTrue Positives (OA correctly identified):")
print(f"  Count: {len(tp_df)}")
print(f"  Mean probability: {tp_df['probability'].mean():.4f}")
print(f"  Confidence > 0.95: {(tp_df['probability'] > 0.95).sum()} ({(tp_df['probability'] > 0.95).sum()/len(tp_df)*100:.2f}%)")
print(f"  Confidence > 0.99: {(tp_df['probability'] > 0.99).sum()} ({(tp_df['probability'] > 0.99).sum()/len(tp_df)*100:.2f}%)")

print(f"\nTrue Negatives (Healthy correctly identified):")
print(f"  Count: {len(tn_df)}")
print(f"  Mean probability: {tn_df['probability'].mean():.4f}")
print(f"  Confidence < 0.05: {(tn_df['probability'] < 0.05).sum()} ({(tn_df['probability'] < 0.05).sum()/len(tn_df)*100:.2f}%)")
print(f"  Confidence < 0.01: {(tn_df['probability'] < 0.01).sum()} ({(tn_df['probability'] < 0.01).sum()/len(tn_df)*100:.2f}%)")

# ============================================================================
# 3. Additional Threshold Testing
# ============================================================================
print("\n" + "="*80)
print("3. CLINICAL SCENARIO THRESHOLD TESTING")
print("="*80)

# Test multiple thresholds
thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_results = []

for thresh in thresholds:
    preds = (df['probability'] >= thresh).astype(int)
    labels = df['label'].values

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    balanced_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    total_errors = fp + fn
    error_rate = total_errors / len(df)

    threshold_results.append({
        'threshold': thresh,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'total_errors': total_errors,
        'error_rate': error_rate,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'balanced_accuracy': balanced_acc,
        'F1': f1,
        'MCC': mcc
    })

threshold_df = pd.DataFrame(threshold_results)

# Print key thresholds
print("\nKey Clinical Scenarios:")
print("-" * 80)

# Scenario 1: Maximize sensitivity (don't miss any OA patients)
max_sensitivity_idx = threshold_df['sensitivity'].idxmax()
print(f"\n1. Maximum Sensitivity (Screening mode - 환자 놓치지 않기):")
print(f"   Threshold: {threshold_df.loc[max_sensitivity_idx, 'threshold']}")
print(f"   Sensitivity: {threshold_df.loc[max_sensitivity_idx, 'sensitivity']:.4f}")
print(f"   Specificity: {threshold_df.loc[max_sensitivity_idx, 'specificity']:.4f}")
print(f"   Total Errors: {threshold_df.loc[max_sensitivity_idx, 'total_errors']:.0f} (FP: {threshold_df.loc[max_sensitivity_idx, 'FP']:.0f}, FN: {threshold_df.loc[max_sensitivity_idx, 'FN']:.0f})")

# Scenario 2: Balanced (minimize total errors)
min_error_idx = threshold_df['total_errors'].idxmin()
print(f"\n2. Minimum Total Errors (Balanced mode):")
print(f"   Threshold: {threshold_df.loc[min_error_idx, 'threshold']}")
print(f"   Sensitivity: {threshold_df.loc[min_error_idx, 'sensitivity']:.4f}")
print(f"   Specificity: {threshold_df.loc[min_error_idx, 'specificity']:.4f}")
print(f"   Total Errors: {threshold_df.loc[min_error_idx, 'total_errors']:.0f} (FP: {threshold_df.loc[min_error_idx, 'FP']:.0f}, FN: {threshold_df.loc[min_error_idx, 'FN']:.0f})")

# Scenario 3: Maximum F1
max_f1_idx = threshold_df['F1'].idxmax()
print(f"\n3. Maximum F1 Score:")
print(f"   Threshold: {threshold_df.loc[max_f1_idx, 'threshold']}")
print(f"   F1 Score: {threshold_df.loc[max_f1_idx, 'F1']:.4f}")
print(f"   Sensitivity: {threshold_df.loc[max_f1_idx, 'sensitivity']:.4f}")
print(f"   Specificity: {threshold_df.loc[max_f1_idx, 'specificity']:.4f}")

# Scenario 4: Default (0.5)
default_idx = threshold_df[threshold_df['threshold'] == 0.5].index[0]
print(f"\n4. Default Threshold (0.5):")
print(f"   Sensitivity: {threshold_df.loc[default_idx, 'sensitivity']:.4f}")
print(f"   Specificity: {threshold_df.loc[default_idx, 'specificity']:.4f}")
print(f"   Total Errors: {threshold_df.loc[default_idx, 'total_errors']:.0f} (FP: {threshold_df.loc[default_idx, 'FP']:.0f}, FN: {threshold_df.loc[default_idx, 'FN']:.0f})")

# Save threshold results
threshold_df.to_csv(OUTPUT_PATH / 'threshold_analysis.csv', index=False)
print(f"\n[OK] Threshold analysis saved: {OUTPUT_PATH / 'threshold_analysis.csv'}")

# ============================================================================
# 4. Visualizations
# ============================================================================
print("\n" + "="*80)
print("4. GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. Probability distribution by error type
ax1 = fig.add_subplot(gs[0, 0])
for error_type in ['TP', 'TN', 'FP', 'FN']:
    subset = df[df['error_type'] == error_type]
    if len(subset) > 0:
        ax1.hist(subset['probability'], bins=30, alpha=0.5, label=error_type)
ax1.set_xlabel('Probability')
ax1.set_ylabel('Count')
ax1.set_title('Probability Distribution by Error Type', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Box plot by error type
ax2 = fig.add_subplot(gs[0, 1])
data_for_box = [df[df['error_type'] == et]['probability'].values
                for et in ['TP', 'TN', 'FP', 'FN']]
ax2.boxplot(data_for_box, labels=['TP', 'TN', 'FP', 'FN'])
ax2.set_ylabel('Probability')
ax2.set_title('Probability Distribution by Error Type', fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

# 3. Error window IDs
ax3 = fig.add_subplot(gs[0, 2])
error_df = df[~df['correct']]
ax3.scatter(error_df['window_id'], error_df['probability'],
           c=['red' if et == 'FN' else 'orange' for et in error_df['error_type']],
           alpha=0.7)
ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Default threshold (0.5)')
ax3.set_xlabel('Window ID')
ax3.set_ylabel('Probability')
ax3.set_title('Error Window Distribution', fontweight='bold')
ax3.legend(['Default threshold', 'FN', 'FP'])
ax3.grid(alpha=0.3)

# 4. Threshold vs Sensitivity/Specificity
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(threshold_df['threshold'], threshold_df['sensitivity'],
        'b-', linewidth=2, label='Sensitivity', marker='o')
ax4.plot(threshold_df['threshold'], threshold_df['specificity'],
        'r-', linewidth=2, label='Specificity', marker='s')
ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
ax4.axvline(x=0.03, color='green', linestyle='--', alpha=0.5, label='Optimal (0.03)')
ax4.set_xlabel('Threshold')
ax4.set_ylabel('Rate')
ax4.set_title('Sensitivity vs Specificity by Threshold', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Threshold vs Total Errors
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(threshold_df['threshold'], threshold_df['total_errors'],
        'purple', linewidth=2, marker='o')
ax5.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax5.axvline(x=0.03, color='green', linestyle='--', alpha=0.5)
ax5.set_xlabel('Threshold')
ax5.set_ylabel('Total Errors')
ax5.set_title('Total Errors by Threshold', fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Threshold vs F1/MCC
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(threshold_df['threshold'], threshold_df['F1'],
        'green', linewidth=2, label='F1 Score', marker='o')
ax6.plot(threshold_df['threshold'], threshold_df['MCC'],
        'blue', linewidth=2, label='MCC', marker='s')
ax6.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel('Threshold')
ax6.set_ylabel('Score')
ax6.set_title('F1 & MCC by Threshold', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 7. FP/FN vs Threshold
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(threshold_df['threshold'], threshold_df['FP'],
        'orange', linewidth=2, label='False Positives', marker='o')
ax7.plot(threshold_df['threshold'], threshold_df['FN'],
        'red', linewidth=2, label='False Negatives', marker='s')
ax7.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax7.axvline(x=0.03, color='green', linestyle='--', alpha=0.5)
ax7.set_xlabel('Threshold')
ax7.set_ylabel('Count')
ax7.set_title('FP vs FN by Threshold', fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. PPV/NPV vs Threshold
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(threshold_df['threshold'], threshold_df['PPV'],
        'cyan', linewidth=2, label='PPV (Precision)', marker='o')
ax8.plot(threshold_df['threshold'], threshold_df['NPV'],
        'magenta', linewidth=2, label='NPV', marker='s')
ax8.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax8.set_xlabel('Threshold')
ax8.set_ylabel('Rate')
ax8.set_title('PPV vs NPV by Threshold', fontweight='bold')
ax8.legend()
ax8.grid(alpha=0.3)

# 9. Balanced Accuracy vs Threshold
ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(threshold_df['threshold'], threshold_df['balanced_accuracy'],
        'darkblue', linewidth=2, marker='o')
ax9.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax9.axvline(x=0.03, color='green', linestyle='--', alpha=0.5)
ax9.set_xlabel('Threshold')
ax9.set_ylabel('Balanced Accuracy')
ax9.set_title('Balanced Accuracy by Threshold', fontweight='bold')
ax9.grid(alpha=0.3)

# 10. Probability scatter: Correct vs Incorrect
ax10 = fig.add_subplot(gs[3, :])
correct_df = df[df['correct']]
incorrect_df = df[~df['correct']]
ax10.scatter(correct_df['window_id'], correct_df['probability'],
            c='green', alpha=0.3, s=20, label='Correct')
ax10.scatter(incorrect_df['window_id'], incorrect_df['probability'],
            c='red', alpha=0.8, s=50, label='Incorrect', marker='x')
ax10.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax10.set_xlabel('Window ID')
ax10.set_ylabel('Probability')
ax10.set_title('All Predictions: Correct vs Incorrect', fontweight='bold')
ax10.legend()
ax10.grid(alpha=0.3)

plt.suptitle('Local Prediction Analysis - Phase 1-3',
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_PATH / 'local_analysis.png', dpi=300, bbox_inches='tight')
print(f"[OK] Visualization saved: {OUTPUT_PATH / 'local_analysis.png'}")
plt.close()

# ============================================================================
# 5. Summary Report
# ============================================================================
print("\n" + "="*80)
print("5. SUMMARY REPORT")
print("="*80)

summary = {
    'total_windows': int(len(df)),
    'oa_windows': int((df['label'] == 1).sum()),
    'healthy_windows': int((df['label'] == 0).sum()),
    'correct_predictions': int(df['correct'].sum()),
    'incorrect_predictions': int((~df['correct']).sum()),
    'error_rate': float((~df['correct']).sum() / len(df)),

    'fn_count': int(len(fn_df)),
    'fn_mean_prob': float(fn_df['probability'].mean()) if len(fn_df) > 0 else 0,
    'fn_std_prob': float(fn_df['probability'].std()) if len(fn_df) > 0 else 0,
    'fn_window_ids': [int(x) for x in sorted(fn_df['window_id'].tolist())],

    'fp_count': int(len(fp_df)),
    'fp_mean_prob': float(fp_df['probability'].mean()) if len(fp_df) > 0 else 0,
    'fp_std_prob': float(fp_df['probability'].std()) if len(fp_df) > 0 else 0,
    'fp_window_ids': [int(x) for x in sorted(fp_df['window_id'].tolist())],

    'tp_count': int(len(tp_df)),
    'tp_mean_prob': float(tp_df['probability'].mean()),
    'tp_high_confidence': int((tp_df['probability'] > 0.99).sum()),

    'tn_count': int(len(tn_df)),
    'tn_mean_prob': float(tn_df['probability'].mean()),
    'tn_high_confidence': int((tn_df['probability'] < 0.01).sum()),

    'best_threshold': float(threshold_df.loc[min_error_idx, 'threshold']),
    'best_threshold_errors': int(threshold_df.loc[min_error_idx, 'total_errors']),
    'best_threshold_sensitivity': float(threshold_df.loc[min_error_idx, 'sensitivity']),
    'best_threshold_specificity': float(threshold_df.loc[min_error_idx, 'specificity']),

    'default_threshold_errors': int(threshold_df.loc[default_idx, 'total_errors']),
    'error_reduction': int(threshold_df.loc[default_idx, 'total_errors'] - threshold_df.loc[min_error_idx, 'total_errors']),
    'error_reduction_pct': float((threshold_df.loc[default_idx, 'total_errors'] - threshold_df.loc[min_error_idx, 'total_errors']) / threshold_df.loc[default_idx, 'total_errors'] * 100)
}

with open(OUTPUT_PATH / 'local_analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[OK] Summary saved: {OUTPUT_PATH / 'local_analysis_summary.json'}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print(f"\n1. Error Distribution:")
print(f"   - Total Errors: {summary['incorrect_predictions']} ({summary['error_rate']*100:.2f}%)")
print(f"   - False Negatives: {summary['fn_count']} (Mean prob: {summary['fn_mean_prob']:.4f})")
print(f"   - False Positives: {summary['fp_count']} (Mean prob: {summary['fp_mean_prob']:.4f})")

print(f"\n2. Threshold Optimization:")
print(f"   - Best Threshold: {summary['best_threshold']}")
print(f"   - Error Reduction: {summary['error_reduction']:.0f} ({summary['error_reduction_pct']:.1f}% improvement)")
print(f"   - Best Performance: {summary['best_threshold_errors']:.0f} errors")
print(f"   - Sensitivity: {summary['best_threshold_sensitivity']:.4f}")
print(f"   - Specificity: {summary['best_threshold_specificity']:.4f}")

print(f"\n3. Confidence Analysis:")
print(f"   - TP High Confidence (>0.99): {summary['tp_high_confidence']}/{summary['tp_count']} ({summary['tp_high_confidence']/summary['tp_count']*100:.1f}%)")
print(f"   - TN High Confidence (<0.01): {summary['tn_high_confidence']}/{summary['tn_count']} ({summary['tn_high_confidence']/summary['tn_count']*100:.1f}%)")

print("\n" + "="*80)
print("[DONE] Local Analysis Complete!")
print(f"Output directory: {OUTPUT_PATH}")
print("="*80)
