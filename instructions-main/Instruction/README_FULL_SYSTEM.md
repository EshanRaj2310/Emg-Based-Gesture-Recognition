# EMG Gesture Recognition System - Complete Implementation

## Overview

This project implements a **fast-adapting gesture recognition system** for EMG (Electromyography) signals. The key innovation is **model adaptation**: instead of retraining from scratch every time the sensor band is repositioned, the system quickly fine-tunes a pre-trained base model in ~30 seconds.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION (azmuth.py)                              │
│    - Collect 7 sessions × 6 gestures @ 200Hz                │
│    - Each session: 45s (15 reps × 3s each)                  │
│    - Output: CSV files in data_raw/                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. BASE MODEL TRAINING (train_base_model.py)                │
│    - CNN + LSTM neural network                              │
│    - 80/20 train/test split                                 │
│    - Learns gesture patterns across sessions                │
│    - Output: base_model_best.pth                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FAST CALIBRATION (calibrate_model.py)                    │
│    - User performs quick cal (3-5s per gesture)             │
│    - Fine-tunes last FC layers only                         │
│    - Takes ~20-30 seconds total                             │
│    - Output: <user>_adapted_model.pth                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. REAL-TIME PREDICTION (predict_realtime.py)               │
│    - Load adapted model                                     │
│    - Stream EMG data from sensor                            │
│    - Predict gestures in real-time                          │
│    - Confidence-based filtering                             │
└─────────────────────────────────────────────────────────────┘
```

## The 6 Gestures

| Gesture | Meaning | ADC Signature |
|---------|---------|---------------|
| **rest** | Neutral / no action | Low baseline |
| **open** | No / Stop | Spread fingers, high spread |
| **fist** | Yes / OK | Light squeeze |
| **strong_fist** | Urgent / Help | Maximum contraction |
| **wrist_up** | Hello / Start | Wrist extension |
| **wrist_down** | Goodbye / End | Wrist flexion |

## Key Features

### ✅ 200Hz Sampling
- **Updated from 5Hz to 200Hz** for better EMG resolution
- 600 samples per 3-second gesture repetition
- Captures fine EMG muscle activation details

### ✅ Multi-Session Training
- 7 sessions per gesture = 42 total recording sessions
- Trains model to be robust across electrode variations
- ~9,000 samples per gesture

### ✅ Fast Calibration (Novel)
- Instead of retraining from scratch: **fine-tune only FC layers**
- Takes 20-30 seconds vs hours for full retraining
- Handles inter-session variability (electrode shift, skin impedance change)

### ✅ CNN + LSTM Architecture
- **CNN layers**: Extract spatial patterns from raw EMG
- **LSTM layers**: Capture temporal dynamics of gesture
- **FC layers**: Fine-tuned during calibration for adaptation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step-by-Step Workflow

### Step 1: Collect Base Dataset (One-Time)

```bash
python azmuth.py
```

This collects 7 sessions of data for all 6 gestures:
- You'll be prompted 6 times (once per gesture)
- Each session: 45 seconds (15 × 3-second repetitions)
- **Total time: ~6 minutes per session × 7 sessions = ~42 minutes**

Output files (in `data_raw/`):
```
data_raw/
├── rest.csv          (7 sessions × 15 reps each = 105 samples)
├── open.csv
├── fist.csv
├── strong_fist.csv
├── wrist_up.csv
└── wrist_down.csv
```

Each CSV contains columns:
- `ms`: Timestamp in milliseconds
- `adc_ch1`: Raw ADC values for sensor 1
- `volt_ch1`: Converted voltage for sensor 1
- `adc_ch2`: Raw ADC values for sensor 2
- `volt_ch2`: Converted voltage for sensor 2
- `rep`: Repetition number (1-15)

**CSV Structure** (example first few rows):
```
ms,adc_ch1,volt_ch1,adc_ch2,volt_ch2,rep
0,512,0.412,478,0.385,1
5,514,0.414,480,0.387,1
10,516,0.415,482,0.388,1
...
45000,498,0.401,465,0.375,15
```

### Step 2: Train Base Model (One-Time)

```bash
python train_base_model.py
```

This trains a CNN+LSTM model on all collected data:

1. **Loads all CSV files** from `data_raw/`
2. **Extracts features**:
   - Time-domain: RMS, mean absolute value, variance, waveform length, zero crossings
   - Frequency-domain: Mean frequency, median frequency, power
3. **Creates CNN+LSTM model**:
   - Conv1D layers (32→64→128 filters)
   - LSTM layers (128→64 hidden units)
   - FC classification head
4. **Trains for 100 epochs** with 80/20 train/test split
5. **Saves best model** to `models/base_model_best.pth`

Output:
```
models/
├── base_model_best.pth       (Best model weights)
├── base_model_final.pth      (Final model weights)
├── gesture_mappings.pkl      (Gesture index mappings)
└── training_history.png      (Training curves)
```

**Expected Results:**
- Test accuracy: 90-98% (high because data is from same user, same day)
- This is expected! Real challenge is inter-session variability

### Step 3: Fast Calibration (Every Time Band Is Put On)

```bash
python calibrate_model.py
```

This adapts the model to your NEW sensor position:

1. **Quick gesture recording**: 3-5 seconds per gesture
2. **Fine-tunes ONLY FC layers**: Feature extractors stay frozen
3. **Takes 20-30 seconds total**

Process:
```
FOR each gesture:
  1. Prepare gesture pose
  2. Press Enter to start recording
  3. Hold gesture for 3 seconds
  4. Release and prepare for next gesture

After all 6 gestures:
  - Model fine-tunes in ~20 seconds
  - Saves adapted model to models/adapted_models/session_1_adapted_model.pth
```

Output:
```
models/adapted_models/
└── session_1_adapted_model.pth   (Adapted for this session)
```

### Step 4: Real-Time Prediction

```bash
python predict_realtime.py
```

Streams EMG data and predicts gestures in real-time:

```
============================================================
🎯 REAL-TIME GESTURE PREDICTION
============================================================
Window size: 100 samples (0.50s)
Stride: 50 samples (0.25s)
Press Ctrl+C to stop

[0.5s] REST            (87.3%)
[1.0s] FIST            (92.1%)
[1.5s] OPEN            (84.6%)
[2.0s] WRIST_UP        (91.2%)
...
```

## Technical Details

### Data Collection (200Hz)

The updated `azmuth.py` now collects data at **200Hz** (was 5Hz before):

```python
SAMPLING_RATE = 200  # Hz
DURATION_PER_REP = 3  # seconds
SAMPLES_PER_REP = 200 * 3 = 600  # samples per repetition
```

Key changes:
1. Added `SAMPLING_RATE` configuration
2. Adjusted plot update interval to 100ms (was 50ms)
3. Added sample count tracking
4. Reports actual achieved sampling rate

### Model Architecture

```
Input: (Batch, 2, 600)  [2 EMG channels, 600 time samples]
    ↓
Conv1D(2→32, kernel=5) + BatchNorm + ReLU + MaxPool
    ↓
Conv1D(32→64, kernel=5) + BatchNorm + ReLU + MaxPool
    ↓
Conv1D(64→128, kernel=5) + BatchNorm + ReLU + MaxPool
    ↓ Transpose to (Batch, TimeSteps, Features)
LSTM(128→64, 2 layers)
    ↓ Take last hidden state
Dense(64→32) + ReLU + Dropout
    ↓
Dense(32→6)  [6 gestures]
    ↓
Output: (Batch, 6)  [logits for each gesture]
```

### Why Fine-Tuning Works

**Inter-Session Problem:**
```
Session 1: Band on, model accurate
Day 2: Band repositioned, model fails
  → Electrode shift
  → Different skin impedance
  → Signal baseline changed
```

**Solution - Fine-Tuning:**
```
Base model learned:        "How to extract gesture features"
                          (CNN + LSTM do this well)

During calibration:        "Adjust output layer for THIS band position"
                          (Only FC layers retrain)

Result: 20-30s adaptation vs retraining model for hours
```

## Important Parameters

### Collection (azmuth.py)
```python
PORT = "COM5"                    # Change to your ESP32 port
BAUD = 115200                    # Serial baud rate
SAMPLING_RATE = 200              # 200Hz data collection (updated!)
DURATION_PER_REP = 3             # 3 seconds per repetition
REPETITIONS = 15                 # 15 reps per session
TOTAL_DURATION = 45              # 45 seconds per gesture per session
```

### Training (train_base_model.py)
```python
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
```

### Calibration (calibrate_model.py)
```python
CALIBRATION_DURATION = 3         # Seconds per gesture (3-5s ok)
CALIBRATION_REPETITIONS = 1      # Just 1 rep for speed
ADAPTATION_EPOCHS = 20           # Quick fine-tuning
ADAPTATION_LR = 0.0001           # Lower LR for fine-tuning
```

### Real-Time (predict_realtime.py)
```python
WINDOW_SIZE = 100                # 0.5s window
WINDOW_STRIDE = 50               # 0.25s stride (4x overlap)
CONFIDENCE_THRESHOLD = 0.6       # Only predict if >60% confident
```

## Directory Structure After Setup

```
Omnitrix-main/
├── azmuth.py                       # Data collection (updated to 200Hz)
├── train_base_model.py             # Train base model
├── calibrate_model.py              # Fast calibration
├── predict_realtime.py             # Real-time prediction
├── requirements.txt                # Dependencies
│
├── data_raw/                       # Raw EMG data (created during collection)
│   ├── rest.csv
│   ├── open.csv
│   ├── fist.csv
│   ├── strong_fist.csv
│   ├── wrist_up.csv
│   └── wrist_down.csv
│
├── graphs/                         # Plots from data collection
│   ├── rest.png
│   ├── open.png
│   └── ...
│
└── models/                         # Trained models
    ├── base_model_best.pth         # Base model (one-time training)
    ├── base_model_final.pth
    ├── gesture_mappings.pkl
    ├── training_history.png
    │
    └── adapted_models/             # Per-session adapted models
        ├── session_1_adapted_model.pth
        ├── session_2_adapted_model.pth
        └── ...
```

## Performance Expectations

### Collection Phase
- **Time**: ~42 minutes for 7 complete sessions
- **Data size**: ~2-3 MB for all CSV files
- **Quality**: Look at PNG plots to verify good signal

### Training Phase
- **Time**: 5-10 minutes on CPU, 1-2 minutes on GPU
- **Accuracy**: 90-98% on test set (same user, same day)
- **Model size**: ~2 MB

### Calibration Phase
- **Time**: ~30 seconds (1 week vs re-collecting data)
- **Accuracy**: ~90% on calibration data
- **Key benefit**: Handles electrode shift automatically

### Real-Time Phase
- **Latency**: ~500ms per prediction (window size)
- **Throughput**: 4 predictions per second (overlapping windows)
- **Accuracy**: Should maintain ~85-95% (depends on calibration quality)

## Troubleshooting

### Serial Connection Issues
```python
# Check available ports
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
for p in ports:
    print(p)

# Update PORT in scripts
PORT = "COM5"  # Change to correct port
```

### Poor Model Accuracy
1. **Check CSV data quality**: Plot the graphs, look for noise
2. **Verify 200Hz sampling**: Check CSV timestamps are evenly spaced
3. **Collect more sessions**: 7 sessions might not be enough; try 10-15
4. **Check gesture consistency**: Perform same gesture the same way each time

### Calibration Not Working
1. **Perform gestures clearly**: No partial movements
2. **Hold each gesture steady**: 3 seconds of consistent contraction
3. **Ensure good electrode contact**: Clean arm, good paste
4. **Verify data collection**: Check calibration data in console

## Next Steps & Extensions

### 1. Multi-User Model
```python
# Train on data from multiple users
# Much more robust but requires more data collection
train_multi_user_model()
```

### 2. Online Learning
```python
# Continuously update model with new predictions
# Better long-term performance
online_learning_loop()
```

### 3. Transfer Learning
```python
# Use pre-trained EMG models from other datasets
# Reduces data collection burden
```

### 4. Cross-Session Evaluation
```python
# Train on sessions 1-6, test on session 7
# Better estimate of real-world performance
```

## Citation & References

- **EMG Feature Extraction**: Phinyomark et al. (2012)
- **CNN for EMG**: Deep Learning in EMG Signal Processing
- **LSTM for Time Series**: Graves & Schmidhuber (2005)
- **Transfer Learning**: Yosinski et al. (2014)

## License

This project is provided as-is for educational and research purposes.

---

**Happy gesture recognition! 🎯**

For questions or issues, check the individual script documentation headers.
