# 📚 Project File Index

## Core Python Scripts

### 1. **azmuth.py** - EMG Data Collection
- **Status:** ✅ MODIFIED (200Hz sampling)
- **Purpose:** Collect EMG data from ESP32 sensor
- **Key Features:**
  - 200Hz sampling rate (upgraded from 5Hz)
  - 7 sessions × 6 gestures × 15 repetitions
  - Real-time plotting
  - Automatic CSV export
- **Output:** 
  - `data_raw/*.csv` - Raw EMG data
  - `graphs/*.png` - Signal plots
- **Time:** ~42 minutes total
- **Entry Point:** `python azmuth.py`

---

### 2. **train_base_model.py** - Model Training
- **Status:** ✅ NEW
- **Purpose:** Train CNN+LSTM model on collected data
- **Key Features:**
  - Loads all gesture CSV files
  - Extracts 16 time/frequency features
  - CNN (Conv1D) + LSTM architecture
  - 100 epochs with validation monitoring
  - Saves best model weights
- **Input:** `data_raw/*.csv` files
- **Output:**
  - `models/base_model_best.pth` - Best weights
  - `models/base_model_final.pth` - Final weights
  - `models/training_history.png` - Training curves
  - `models/gesture_mappings.pkl` - Gesture indices
- **Time:** 5-10 minutes (CPU), 1-2 minutes (GPU)
- **Expected Accuracy:** 90-98%
- **Entry Point:** `python train_base_model.py`

---

### 3. **calibrate_model.py** - Fast Calibration
- **Status:** ✅ NEW (Solves inter-session variability!)
- **Purpose:** Adapt model to new electrode position (30 seconds)
- **Key Features:**
  - Collects quick calibration data (3-5s per gesture)
  - Freezes CNN+LSTM layers (feature extractors)
  - Fine-tunes only FC layers (2K out of 155K parameters)
  - Ultra-fast adaptation
- **Input:** 
  - `models/base_model_best.pth` - Pre-trained model
  - Serial EMG data from ESP32
- **Output:**
  - `models/adapted_models/session_X_adapted_model.pth` - Adapted model
- **Time:** ~30 seconds total (1 min with UI)
- **When to Run:** Every time band is repositioned
- **Entry Point:** `python calibrate_model.py`

---

### 4. **predict_realtime.py** - Real-Time Inference
- **Status:** ✅ NEW
- **Purpose:** Perform live gesture recognition
- **Key Features:**
  - Streams 200Hz EMG data from serial
  - Sliding window processing (0.5s window, 0.25s stride)
  - Confidence-based filtering (>60% threshold)
  - Real-time console output
  - 4 predictions per second
- **Input:**
  - `models/adapted_models/session_X_adapted_model.pth` - Trained model
  - Serial EMG stream @ 200Hz
- **Output:** Console: Gesture name + confidence
- **Time:** Live (runs continuously)
- **Entry Point:** `python predict_realtime.py`

---

## Documentation Files

### 1. **README_FULL_SYSTEM.md** - Complete System Guide
- **Audience:** Technical users, developers
- **Content:**
  - Full system architecture
  - Detailed workflow explanation
  - Data specifications and formats
  - Model architecture details
  - Feature extraction methods
  - Performance expectations
  - Directory structure
  - Parameter documentation
  - Troubleshooting section
  - Next steps and extensions
- **Length:** ~500 lines
- **When to Read:** Need deep understanding of system

---

### 2. **QUICKSTART.md** - 4-Step Getting Started
- **Audience:** First-time users, developers wanting quick start
- **Content:**
  - TL;DR quick start (4 steps)
  - What's different from original
  - Sampling rate explanation
  - Step-by-step instructions
  - Expected results
  - Common questions (FAQ)
  - Typical session timing
- **Length:** ~200 lines
- **When to Read:** Just want to get running fast

---

### 3. **ARCHITECTURE_DETAILED.md** - Data Flow Visualization
- **Audience:** Visual learners, system designers
- **Content:**
  - Complete ASCII pipeline diagrams
  - Phase-by-phase data transformations
  - CSV format examples with actual data
  - Model architecture visualization
  - Sliding window explanation
  - Concept: Why fast calibration works
  - Data flow examples
- **Length:** ~400 lines
- **When to Read:** Understand the "big picture"

---

### 4. **IMPLEMENTATION_SUMMARY.md** - What Was Built
- **Audience:** Project stakeholders, summary readers
- **Content:**
  - What was implemented (checkmarks)
  - System specifications
  - Complete workflow
  - Key improvements
  - Directory structure
  - Technical architecture
  - Model parameters
  - Next steps
- **Length:** ~300 lines
- **When to Read:** See what's included in this project

---

### 5. **TROUBLESHOOTING.md** - Problem Solving Guide
- **Audience:** Users encountering issues
- **Content:**
  - 9 categories of common issues:
    1. Serial connection issues
    2. Data collection issues
    3. Model training issues
    4. Calibration issues
    5. Real-time prediction issues
    6. Performance issues
    7. Accuracy & reliability issues
    8. Data file issues
    9. Installation issues
  - Step-by-step solutions
  - Debug code examples
  - Quick diagnostic checklist
- **Length:** ~400 lines
- **When to Read:** Something isn't working

---

### 6. **PROJECT_COMPLETION.md** - Completion Summary
- **Audience:** Project overview, stakeholders
- **Content:**
  - What was asked for
  - What was delivered
  - System specifications
  - Complete workflow
  - Key innovations
  - Final directory structure
  - How to use
  - Documentation map
- **Length:** ~300 lines
- **When to Read:** Get project overview

---

## Configuration Files

### **requirements.txt** - Python Dependencies
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-learn>=0.24.0
torch>=1.10.0
torchvision>=0.11.0
pyserial>=3.5
tqdm>=4.62.0
joblib>=1.1.0
```

**Install:** `pip install -r requirements.txt`

---

## Data Directories (Created During Runtime)

### **data_raw/** - Raw EMG Data (CSV files)
- Created by: `azmuth.py`
- Contains: 6 CSV files (one per gesture)
- Format: ms, adc_ch1, volt_ch1, adc_ch2, volt_ch2, rep
- Size per file: ~9,000 rows (600 samples × 15 reps)
- Total size: ~2-3 MB

---

### **graphs/** - Visualization Plots (PNG files)
- Created by: `azmuth.py`
- Contains: 6 PNG files (one per gesture)
- Shows: 2-subplot plots of 45-second EMG recordings
- Resolution: 200 DPI
- Total size: ~100-200 KB

---

### **models/** - Trained Models
- Created by: `train_base_model.py` and `calibrate_model.py`
- Structure:
  ```
  models/
  ├── base_model_best.pth           (best weights during training)
  ├── base_model_final.pth          (final weights after training)
  ├── gesture_mappings.pkl          (gesture↔index mapping)
  ├── training_history.png          (loss & accuracy curves)
  └── adapted_models/               (per-session adapted models)
      ├── session_1_adapted_model.pth
      ├── session_2_adapted_model.pth
      └── ...
  ```
- Model file size: ~2 MB each

---

## Reading Order (Recommended)

### First Time Setup:
1. **PROJECT_COMPLETION.md** - Understand what's included
2. **QUICKSTART.md** - Get running in 4 steps
3. Run `python azmuth.py` - Start collecting data

### During Training:
1. **README_FULL_SYSTEM.md** - Deep technical understanding
2. **ARCHITECTURE_DETAILED.md** - Visual understanding of data flow

### When Issues Arise:
1. **TROUBLESHOOTING.md** - Problem solver

### Deep Dive:
1. **ARCHITECTURE_DETAILED.md** - Complete system design
2. Script docstrings and comments - Implementation details

---

## Quick Reference

### The 4-Step Workflow
```
1. python azmuth.py               → data_raw/*.csv
2. python train_base_model.py     → models/base_model_best.pth
3. python calibrate_model.py      → models/adapted_models/session_X_adapted_model.pth
4. python predict_realtime.py     → Live gesture recognition
```

### Time Requirements
- Step 1: 42 minutes (7 sessions)
- Step 2: 10 minutes (training)
- Step 3: 30 seconds (calibration)
- Step 4: Continuous

### Key Numbers
- Sampling: **200 Hz** (samples per second)
- Gestures: **6** (rest, open, fist, strong_fist, wrist_up, wrist_down)
- Sessions: **7** per gesture
- Repetitions: **15** per session
- Duration per rep: **3 seconds**
- Samples per rep: **600** (200 Hz × 3 s)
- Total samples per gesture: **9,000** (15 × 600)
- Model parameters: **155,000** (mostly CNN+LSTM)
- Calibration parameters: **2,000** (only FC layers)
- Predictions per second: **4** (0.25s stride)

---

## File Statistics

| Category | Count | Purpose |
|----------|-------|---------|
| **Python Scripts** | 4 | Core functionality |
| **Documentation Files** | 6 | Guides & references |
| **Config Files** | 1 | Dependencies |
| **Data Directories** | 3 | Runtime data |

**Total Python LOC:** ~1,500 lines of code
**Total Documentation:** ~2,000 lines
**Complete Package Size:** ~50-100 MB (including data + models)

---

## Version Control Note

**Original Files Modified:**
- ✏️ `azmuth.py` - Updated to 200Hz sampling

**New Files Added:**
- ✨ `train_base_model.py`
- ✨ `calibrate_model.py`
- ✨ `predict_realtime.py`
- ✨ `README_FULL_SYSTEM.md`
- ✨ `QUICKSTART.md`
- ✨ `ARCHITECTURE_DETAILED.md`
- ✨ `IMPLEMENTATION_SUMMARY.md`
- ✨ `TROUBLESHOOTING.md`
- ✨ `PROJECT_COMPLETION.md`
- ✨ `requirements.txt` (updated)

---

## Need Help?

| Situation | Check This |
|-----------|-----------|
| Want quick start | `QUICKSTART.md` |
| Understand the system | `README_FULL_SYSTEM.md` |
| See data flow | `ARCHITECTURE_DETAILED.md` |
| Something broken | `TROUBLESHOOTING.md` |
| See what's included | `PROJECT_COMPLETION.md` |
| Running script | Script docstring and inline comments |

---

**Good luck with your EMG gesture recognition system! 🎯**

*This index file helps you navigate the complete project.*
