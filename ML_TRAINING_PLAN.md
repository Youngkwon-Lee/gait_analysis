# ML Training Plan: Gait Analysis Classification

## 1. Dataset Overview

### 1.1 Data Statistics
| Cohort | Group | Subjects | Trials | Class |
|--------|-------|----------|--------|-------|
| HS | healthy | 73 | 360 | 0 (Healthy) |
| PD | neuro | 24 | 160 | 1 (Neuro) |
| CVA | neuro | 49 | 128 | 1 (Neuro) |
| CIPN | neuro | 19 | 98 | 1 (Neuro) |
| RIL | neuro | 51 | 398 | 1 (Neuro) |
| HOA | ortho | 15 | 74 | 2 (Ortho) |
| KOA | ortho | 18 | 78 | 2 (Ortho) |
| ACL | ortho | 11 | 60 | 2 (Ortho) |

**Total**: 260 subjects, 1356 trials

### 1.2 Class Distribution
- Healthy: 360 trials (26.5%)
- Neuro: 784 trials (57.8%)
- Ortho: 212 trials (15.6%)

**Imbalance Issue**: Significant class imbalance - needs addressing

---

## 2. Proposed Classification Tasks

### Task 1: Binary Classification (Healthy vs Pathological)
- **Classes**: Healthy (360) vs Pathological (996)
- **Expected Baseline**: ~73% (majority class)
- **Target**: >85% balanced accuracy

### Task 2: 3-Class Classification (Healthy vs Neuro vs Ortho)
- **Classes**: Healthy (360), Neuro (784), Ortho (212)
- **Expected Baseline**: ~58% (majority class)
- **Target**: >75% balanced accuracy

### Task 3: Disease-Specific Screening
#### 3a. PD Screening (Healthy vs PD)
- Healthy: 360, PD: 160
- Clinical relevance: Early Parkinson's detection

#### 3b. Stroke Detection (Healthy vs CVA)
- Healthy: 360, CVA: 128
- Clinical relevance: Post-stroke gait assessment

#### 3c. OA Screening (Healthy vs HOA+KOA)
- Healthy: 360, OA: 152
- Clinical relevance: Osteoarthritis detection

---

## 3. Data Preprocessing Pipeline

### 3.1 Raw Signal Processing
```
Input: 4 sensors × 9 channels × variable length
       (HE, LB, LF, RF) × (Acc_XYZ, Gyr_XYZ, Mag_XYZ)

Processing Steps:
1. Resample to 100Hz (if needed)
2. Remove first/last 2 seconds (standing phases)
3. Bandpass filter: 0.5-25Hz (Butterworth 4th order)
4. Normalize per-sensor (z-score)
5. Segment into fixed windows (2-5 seconds)
```

### 3.2 Feature Engineering Options

#### Option A: Handcrafted Features (Traditional ML)
```python
features = {
    # Temporal
    'stride_time_mean', 'stride_time_std', 'stride_time_cv',
    'swing_time_mean', 'stance_time_mean',
    'cadence', 'walking_speed_proxy',

    # Spatial (from accelerometer)
    'step_regularity', 'stride_regularity',
    'acc_rms_vertical', 'acc_range_vertical',

    # Spectral
    'dominant_freq_gyr_y', 'spectral_entropy',
    'power_ratio_0.5-3Hz',

    # Asymmetry
    'left_right_asymmetry', 'swing_asymmetry',

    # Complexity
    'sample_entropy', 'lyapunov_exponent',
}
# Total: ~50-100 features
```

#### Option B: Deep Learning (End-to-End)
```
Input Shape: (batch, 4_sensors, 9_channels, time_steps)
            = (batch, 4, 9, 500) for 5-second windows

Architectures:
1. 1D-CNN per sensor → Attention Fusion
2. LSTM/GRU with sensor attention
3. Transformer encoder
4. TCN (Temporal Convolutional Network)
```

---

## 4. Model Architecture Recommendations

### 4.1 Baseline Models (Traditional ML)
```
1. Random Forest (n_estimators=200)
2. XGBoost (with class weights)
3. SVM (RBF kernel, with SMOTE)
4. Logistic Regression (L2 regularization)
```

### 4.2 Deep Learning Models

#### Architecture 1: Multi-Stream CNN (Baseline Paper)
```
[4× 1D-CNN branches] → Sensor Attention → FC → Classification
- Conv1D(64, kernel=7) → BN → ReLU → MaxPool
- Conv1D(128, kernel=5) → BN → ReLU → MaxPool
- Conv1D(256, kernel=3) → BN → ReLU → GlobalAvgPool
- Attention weights (learned per sensor)
- FC(256) → Dropout(0.5) → FC(num_classes)
```

#### Architecture 2: LSTM with Attention
```
[4× Bi-LSTM] → Multi-Head Attention → FC → Classification
- Bi-LSTM(128, return_sequences=True)
- Multi-Head Attention(heads=4)
- GlobalAvgPool + GlobalMaxPool
- FC(256) → Dropout(0.5) → FC(num_classes)
```

#### Architecture 3: Transformer Encoder
```
[Linear Projection] → Positional Encoding → Transformer → FC
- Patch embedding (sensor × channel as tokens)
- 4 Transformer encoder layers
- CLS token for classification
```

---

## 5. Training Strategy

### 5.1 Data Split (Subject-Wise)
```
CRITICAL: Split by SUBJECT, not by trial!

Train: 70% subjects (~182 subjects)
Val:   15% subjects (~39 subjects)
Test:  15% subjects (~39 subjects)

Stratified by cohort to maintain class distribution
```

### 5.2 Cross-Validation
```
5-Fold Subject-Stratified CV
- Each fold: 20% subjects held out
- Stratify by cohort
- Report mean ± std across folds
```

### 5.3 Class Imbalance Handling
```
1. Weighted Loss: class_weight = n_samples / (n_classes × n_samples_per_class)
2. Oversampling: SMOTE for traditional ML
3. Data Augmentation for DL:
   - Time warping
   - Magnitude warping
   - Jittering
   - Window slicing
```

### 5.4 Hyperparameter Search
```
Learning Rate: [1e-4, 5e-4, 1e-3]
Batch Size: [16, 32, 64]
Dropout: [0.3, 0.5, 0.7]
Window Size: [200, 300, 500] samples
Optimizer: AdamW with weight decay 1e-4
Scheduler: CosineAnnealing or OneCycleLR
```

---

## 6. Evaluation Metrics

### Primary Metrics
```
1. Balanced Accuracy (macro-averaged recall)
2. ROC-AUC (one-vs-rest for multiclass)
3. PR-AUC (important for imbalanced data)
```

### Secondary Metrics
```
4. Sensitivity (per class)
5. Specificity (per class)
6. F1-Score (macro)
7. Matthews Correlation Coefficient (MCC)
```

### Clinical Metrics
```
8. Confusion Matrix with clinical interpretation
9. Calibration curves (reliability diagrams)
10. Decision curve analysis
```

---

## 7. Implementation Roadmap

### Phase 1: Data Pipeline (Week 1)
- [ ] Load and validate all trials
- [ ] Implement preprocessing pipeline
- [ ] Feature extraction (handcrafted)
- [ ] Subject-wise train/val/test split
- [ ] Data augmentation functions

### Phase 2: Baseline Models (Week 2)
- [ ] Random Forest baseline
- [ ] XGBoost with hyperparameter tuning
- [ ] Establish baseline metrics
- [ ] Feature importance analysis

### Phase 3: Deep Learning (Week 3-4)
- [ ] Implement Multi-Stream CNN
- [ ] Implement LSTM with Attention
- [ ] Training with early stopping
- [ ] Hyperparameter optimization

### Phase 4: Evaluation & Analysis (Week 5)
- [ ] Cross-validation evaluation
- [ ] Sensor importance analysis
- [ ] Error analysis (which cohorts confused)
- [ ] Ablation studies

### Phase 5: Clinical Validation (Week 6)
- [ ] Per-disease performance analysis
- [ ] Comparison with baseline paper
- [ ] Model interpretability (GradCAM, SHAP)
- [ ] Report generation

---

## 8. Expected Results (Based on Baseline Paper)

| Task | Metric | Baseline | Target |
|------|--------|----------|--------|
| PD Screening | ROC-AUC | 0.82 | >0.85 |
| OA Screening | ROC-AUC | 0.99 | >0.95 |
| CVA Detection | ROC-AUC | 0.95 | >0.90 |
| 3-Class | Balanced Acc | ~60% | >75% |

### Known Challenges
1. **Laterality Bias**: OA/CVA cohorts have right-side bias
2. **Age Confound**: PD patients are older
3. **Sensor Type**: XSens vs TechnoConcept differences
4. **Small Ortho Cohort**: Only 212 trials

---

## 9. Code Structure

```
gait_wearable_sensor/
├── data/
│   └── dataset/
├── src/
│   ├── preprocessing/
│   │   ├── loader.py
│   │   ├── filters.py
│   │   └── augmentation.py
│   ├── features/
│   │   ├── temporal.py
│   │   ├── spectral.py
│   │   └── complexity.py
│   ├── models/
│   │   ├── baseline_ml.py
│   │   ├── cnn_multistream.py
│   │   ├── lstm_attention.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── metrics.py
│   └── evaluation/
│       ├── evaluate.py
│       └── visualize.py
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline.ipynb
│   └── 03_DeepLearning.ipynb
├── configs/
│   └── config.yaml
└── outputs/
    ├── models/
    └── results/
```

---

## 10. References

1. Voisard et al. (2025) "A Dataset of Clinical Gait Signals with Wearable Sensors" - Nature Scientific Data
2. "Attention-Based Sensor Optimization and Automated Dataset Auditing" - arXiv 2511.02047
3. PhysioNet gait databases
4. TULIP Project (Timed Up and Go)

---

## Quick Start Commands

```bash
# 1. Setup environment
cd D:\gait_wearable_sensor
python -m venv venv
source venv/Scripts/activate  # Windows
pip install torch numpy pandas scikit-learn matplotlib seaborn

# 2. Run baseline
python src/train_baseline.py --task binary --model rf

# 3. Run deep learning
python src/train_dl.py --model cnn --epochs 100 --batch_size 32
```
