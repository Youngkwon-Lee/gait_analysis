# Directory Structure - Gait Analysis Project

## Overview

This document maps the directory structure across different environments:
- **Local (Windows)**: Development and result review
- **VM (Linux)**: Model training and analysis execution
- **GitHub**: Code repository (data/models/results excluded)

---

## Local Environment (Windows)

```
D:\gait_wearable_sensor\
├── src/                          # Source code
│   ├── train_baseline_hpc.py     # Model training
│   ├── analyze_errors.py         # Error analysis (Phase 1-1)
│   ├── analyze_confusion.py      # Confusion analysis (Phase 1-2)
│   └── ...
├── dataset/                      # Dataset (NOT in Git)
│   └── data/                     # Clinical Gait Signals Dataset
│       ├── Pathological/         # OA patients
│       │   ├── P001/
│       │   │   ├── _raw_data_L-ANKLE.txt
│       │   │   ├── _raw_data_L-FOOT.txt
│       │   │   ├── _raw_data_R-ANKLE.txt
│       │   │   └── _raw_data_R-FOOT.txt
│       │   └── ...
│       └── Healthy/              # Healthy controls
│           └── ...
├── models/                       # Trained models (NOT in Git)
│   ├── OA_Screening_best.pt      # PyTorch checkpoint
│   └── OA_Screening_best.pth     # Model weights
├── results/                      # Analysis results (PNG/CSV NOT in Git)
│   ├── error_analysis/
│   │   ├── OA_Screening_error_analysis.json
│   │   └── OA_Screening_error_analysis.png
│   └── confusion_analysis/
│       ├── OA_Screening_confusion_analysis.json
│       └── OA_Screening_confusion_analysis.png
├── PHASE1_RESULTS.md             # Phase 1 documentation
├── DIRECTORY_STRUCTURE.md        # This file
├── README.md                     # Project overview
└── .gitignore                    # Git exclusions

Total Size:
- dataset/: ~7.4GB
- models/: ~50MB
- results/: ~2MB
```

---

## VM Environment (Linux)

```
/home2/gun3856/gait_code/
├── src/                          # Source code (from GitHub)
├── dataset/                      # Dataset (copied separately)
│   └── data/
│       ├── Pathological/
│       └── Healthy/
├── models/                       # Trained models
│   ├── OA_Screening_best.pt
│   └── OA_Screening_best.pth
├── results/                      # Analysis results
│   ├── error_analysis/
│   └── confusion_analysis/
└── ...

Environment Variables:
- DATA_PATH: $HOME/gait_code/dataset/data
- MODEL_PATH: $HOME/gait_code/models
```

**Note**: VM path may differ. The script writes to `D:/gait_wearable_sensor/` which may be mapped differently on Linux.

---

## GitHub Repository

**Repository**: https://github.com/Youngkwon-Lee/gait_analysis

```
GitHub (Code only):
├── src/                          # All source code ✅
├── PHASE1_RESULTS.md             # Documentation ✅
├── DIRECTORY_STRUCTURE.md        # This file ✅
├── README.md                     # Project overview ✅
├── .gitignore                    # Exclusions ✅
└── requirements.txt              # Dependencies ✅

NOT in GitHub (.gitignore):
├── dataset/                      # Too large (7.4GB)
├── models/                       # Too large (50MB)
├── results/*.png                 # Binary files
└── results/*.csv                 # Generated outputs
```

---

## File Transfer Workflow

### Local ↔ VM

```bash
# Upload code (use Git instead)
git push origin main              # Local → GitHub
git pull origin main              # VM ← GitHub

# Upload data (one-time)
scp -r D:\gait_wearable_sensor\dataset\data gun3856@VM:/home2/gun3856/gait_code/dataset/

# Upload model (if needed)
scp D:\gait_wearable_sensor\models\*.pth gun3856@VM:/home2/gun3856/gait_code/models/

# Download results (after analysis)
scp gun3856@VM:/home2/gun3856/gait_code/results/error_analysis/* C:\Users\YK\Downloads\
```

### Downloads → Project Folder

```bash
# Move downloaded results to project
mv C:\Users\YK\Downloads\OA_Screening_*.json D:\gait_wearable_sensor\results\error_analysis\
mv C:\Users\YK\Downloads\OA_Screening_*.png D:\gait_wearable_sensor\results\error_analysis\
```

---

## Path Configuration

### analyze_errors.py (Line 35-42)

```python
class Config:
    # Use environment variables for cross-platform compatibility
    BASE_PATH = Path(os.environ.get('DATA_PATH', 'D:/gait_wearable_sensor/dataset/data'))
    OUTPUT_PATH = Path(os.environ.get('OUTPUT_PATH', 'D:/gait_wearable_sensor/results/error_analysis'))
    MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'D:/gait_wearable_sensor/models'))
```

**VM Usage**:
```bash
export DATA_PATH="$HOME/gait_code/dataset/data"
export MODEL_PATH="$HOME/gait_code/models"
export OUTPUT_PATH="$HOME/gait_code/results/error_analysis"
python src/analyze_errors.py
```

---

## Dataset Details

### Clinical Gait Signals Dataset
- **Source**: Nature Scientific Data 2025
- **Size**: 7.4GB
- **Subjects**: 179 (81 Healthy, 98 Pathological)
- **Trials**: 800 total
- **Sensors**: 4 IMU sensors (L-ANKLE, L-FOOT, R-ANKLE, R-FOOT)
- **Channels**: 9 per sensor (acc_x/y/z, gyr_x/y/z, mag_x/y/z)
- **Format**: TXT files with header row

### Data Location (TO BE CONFIRMED)

**Local**:
- ✅ Confirmed: `D:\gait_wearable_sensor\dataset\data\`

**VM**:
- ❓ To confirm: `~/gait_code/dataset/data/` OR separate location?
- ❓ Check if dataset was uploaded to VM

**TODO**: Run on VM to confirm dataset location
```bash
find ~ -type d -name "Pathological" -o -name "Healthy" 2>/dev/null
du -sh ~/gait_code/dataset/ 2>/dev/null
```

---

## Analysis Results Tracking

| Analysis | Status | Local Path | VM Path |
|----------|--------|------------|---------|
| Phase 1-1: Error Analysis (old) | ✅ Complete | results/error_analysis/ | ❓ |
| Phase 1-1: Error Analysis (with detailed_predictions) | 🔄 In Progress | - | ❓ Finding output location |
| Phase 1-2: Confusion Analysis | ✅ Complete | results/confusion_analysis/ | ✅ ~/gait_code/results/confusion_analysis/ |

---

## Next Steps

1. ✅ Document directory structure
2. 🔄 Find VM output location for updated Error Analysis
3. ⏭️ Download updated results (with detailed_predictions)
4. ⏭️ Verify detailed_predictions in JSON (575 windows)
5. ⏭️ Plan Phase 2 analyses

---

## Notes

- Always use Git for code synchronization
- Use SCP for data/models/results transfer
- VM output path may need correction in Config class
- Consider using relative paths or environment variables for cross-platform compatibility
