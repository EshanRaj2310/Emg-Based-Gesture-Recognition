# Data Flow & Architecture Visualization

## Complete System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EMG GESTURE RECOGNITION SYSTEM                         │
│                         (200Hz, 6 Gestures)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA COLLECTION (One-time: ~42 minutes)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ESP32 with 2 EMG Sensors                                                   │
│  (Sensors on forearm at pins 34 & 35)                                       │
│         │                                                                    │
│         │ Serial @ 115200 baud                                              │
│         ↓                                                                    │
│  Python: azmuth.py                                                          │
│  ├─ Port: COM5                                                              │
│  ├─ Sampling: 200 Hz (=5ms intervals)                                       │
│  ├─ Duration: 45s per gesture (15 reps × 3s)                               │
│  └─ Gestures: rest, open, fist, strong_fist, wrist_up, wrist_down         │
│         │                                                                    │
│         │ Collect 7 sessions per gesture                                    │
│         │ 15 reps/session × 200 Hz × 3s = 600 samples/rep                 │
│         ↓                                                                    │
│  CSV Files: data_raw/                                                       │
│  ├─ rest.csv          (600 samples × 15 reps × 7 sessions = 63,000 rows)  │
│  ├─ open.csv                                                                │
│  ├─ fist.csv                                                                │
│  ├─ strong_fist.csv                                                         │
│  ├─ wrist_up.csv                                                            │
│  └─ wrist_down.csv                                                          │
│         │                                                                    │
│         └─→ Plots: graphs/ (PNG images per gesture)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: MODEL TRAINING (One-time: ~10 minutes)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: CSV Files (data_raw/*.csv)                                          │
│         │                                                                    │
│         ├─→ Load all 6 gesture files                                        │
│         │   (Total: 630 repetitions × 600 samples = ~378K samples)         │
│         │                                                                    │
│         ├─→ Feature Extraction (per 3s repetition):                         │
│         │   Time-domain: RMS, MAV, Variance, Waveform Length, Zero Cross  │
│         │   Freq-domain: Mean Freq, Median Freq, Power                    │
│         │   Result: 16 features × 2 channels = 32D per sample             │
│         │                                                                    │
│         ├─→ Normalize & Reshape:                                           │
│         │   (630, 2, 600) ← 630 reps, 2 channels, 600 samples each       │
│         │                                                                    │
│         ├─→ Train/Val/Test Split:                                          │
│         │   Train: 60% (378 reps)                                          │
│         │   Val:   15% (94 reps)                                           │
│         │   Test:  25% (158 reps)                                          │
│         │                                                                    │
│         ├─→ Model Architecture:                                             │
│         │                                                                    │
│         │   ┌─────────────────────────────┐                                │
│         │   │ Input (B, 2, 600)           │  B=batch size                  │
│         │   └──────────────┬──────────────┘                                │
│         │                  │                                                │
│         │   ┌──────────────▼──────────────┐                                │
│         │   │ Conv1D(2→32, k=5)           │  Extract spatial patterns      │
│         │   │ BatchNorm + ReLU + MaxPool  │                                │
│         │   └──────────────┬──────────────┘                                │
│         │                  │                                                │
│         │   ┌──────────────▼──────────────┐                                │
│         │   │ Conv1D(32→64, k=5)          │                                │
│         │   │ BatchNorm + ReLU + MaxPool  │                                │
│         │   └──────────────┬──────────────┘                                │
│         │                  │                                                │
│         │   ┌──────────────▼──────────────┐                                │
│         │   │ Conv1D(64→128, k=5)         │                                │
│         │   │ BatchNorm + ReLU + MaxPool  │                                │
│         │   └──────────────┬──────────────┘                                │
│         │                  │                                                │
│         │   ┌──────────────▼──────────────┐                                │
│         │   │ LSTM(128 → 64)              │  Temporal dynamics             │
│         │   │ 2 layers, bidirectional     │                                │
│         │   └──────────────┬──────────────┘                                │
│         │                  │                                                │
│         │   ┌──────────────▼──────────────┐                                │
│         │   │ FC(64 → 32)                 │  Classification head           │
│         │   │ ReLU + Dropout              │                                │
│         │   └──────────────┬──────────────┘                                │
│         │                  │                                                │
│         │   ┌──────────────▼──────────────┐                                │
│         │   │ FC(32 → 6)                  │  6 gestures                    │
│         │   │ Softmax                     │                                │
│         │   └──────────────┬──────────────┘                                │
│         │                  │                                                │
│         │   ┌──────────────▼──────────────┐                                │
│         │   │ Output (B, 6)               │  Confidence per gesture        │
│         │   └──────────────────────────────┘                                │
│         │                                                                    │
│         ├─→ Training (100 epochs):                                          │
│         │   Optimizer: Adam (LR=0.001)                                     │
│         │   Loss: CrossEntropy                                             │
│         │   Scheduler: Cosine Annealing                                    │
│         │                                                                    │
│         ├─→ Best Model Monitoring:                                          │
│         │   Save when Val Accuracy improves                                │
│         │   (Expected: 90-98% test accuracy)                               │
│         │                                                                    │
│         └─→ Output: models/base_model_best.pth                             │
│            Also: training_history.png, gesture_mappings.pkl                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: FAST CALIBRATION (Per Session: ~30 seconds)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ⚠️ Problem: Band repositioned = electrode shift = model accuracy drops     │
│                                                                              │
│  Solution: Load base model + quick fine-tune on new position               │
│                                                                              │
│  Input: Base Model (base_model_best.pth)                                    │
│         │                                                                    │
│         ├─→ Load model                                                      │
│         │                                                                    │
│         ├─→ Collect Quick Cal Data (3-5s per gesture):                     │
│         │   Just 1 repetition per gesture (vs 7×15 before)                 │
│         │   6 gestures × 3s = 18s collection time                          │
│         │   Input: (6, 2, 600) ← 6 gestures × 2 channels × 600 samples    │
│         │                                                                    │
│         ├─→ ❄️ FREEZE Feature Extractors:                                   │
│         │   │ Conv1D layers      │ FROZEN                                  │
│         │   │ LSTM layers        │ FROZEN                                  │
│         │   Params frozen: ~150K                                           │
│         │                                                                    │
│         ├─→ 🔥 UNFREEZE Classification Head:                                │
│         │   │ FC(64→32)          │ TRAINABLE                               │
│         │   │ FC(32→6)           │ TRAINABLE                               │
│         │   Params trainable: ~2K                                          │
│         │                                                                    │
│         ├─→ Quick Training (20 epochs):                                     │
│         │   Optimizer: Adam (LR=0.0001) ← Lower LR!                       │
│         │   Loss: CrossEntropy                                             │
│         │   Expected Cal Accuracy: ~95-98%                                 │
│         │                                                                    │
│         └─→ Output: models/adapted_models/session_1_adapted_model.pth      │
│                                                                              │
│  Why This Works:                                                            │
│  ✓ Feature extractors (CNN+LSTM) already learned "how to see" EMG         │
│  ✓ Only need to adjust output layer for NEW electrode position            │
│  ✓ 99% parameter efficiency vs full retraining                            │
│  ✓ Takes 30 seconds instead of 42 min data + 10 min training              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: REAL-TIME PREDICTION (Live inference)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Adapted Model + Streaming EMG Data                                  │
│         │                                                                    │
│         ├─→ Stream from Serial (200 Hz = 5ms intervals):                   │
│         │   Sensor 1 (pin 34) → ADC value                                  │
│         │   Sensor 2 (pin 35) → ADC value                                  │
│         │                                                                    │
│         ├─→ Sliding Window Processing:                                      │
│         │   Window Size: 100 samples (0.5 seconds)                         │
│         │   Stride: 50 samples (0.25 seconds)  ← 4x overlap                │
│         │   New prediction every 0.25s                                      │
│         │                                                                    │
│         ├─→ Per-Window Processing:                                          │
│         │   Input buffer: [ch1_100, ch2_100]                              │
│         │   ↓                                                               │
│         │   Normalize: (value - mean) / std                                │
│         │   ↓                                                               │
│         │   Model forward: (2, 100) → (6,) logits                         │
│         │   ↓                                                               │
│         │   Softmax: logits → probabilities                                │
│         │   ↓                                                               │
│         │   Argmax: probs → predicted gesture                              │
│         │                                                                    │
│         ├─→ Confidence Filtering:                                           │
│         │   Only output if confidence > 60%                                │
│         │   Prevents false positives from noise                            │
│         │                                                                    │
│         ├─→ Smoothing (optional):                                           │
│         │   Keep history of last 5 predictions                             │
│         │   Majority vote + average confidence                             │
│         │   Reduces jitter from high-frequency noise                       │
│         │                                                                    │
│         └─→ Real-Time Output:                                               │
│            [0.5s] REST            (87.3%)                                  │
│            [1.0s] FIST            (92.1%)                                  │
│            [1.5s] OPEN            (84.6%)                                  │
│            [2.0s] WRIST_UP        (91.2%)                                  │
│            ...                                                              │
│                                                                              │
│  Latency: ~500ms per prediction (one window)                                │
│  Throughput: 4 predictions/sec (overlapping windows)                        │
│  Accuracy: 85-95% (depends on calibration quality)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


COMPLETE DATA FLOW EXAMPLE
═════════════════════════════════════════════════════════════════════════════

Day 1 - Initial Setup:
────────────────────
 azmuth.py
   ↓ Collect rest gesture × 7 sessions
   → 105 repetitions × 600 samples each = 63,000 rows
   → rest.csv (63,000 rows)
   ↓ Collect open, fist, strong_fist, wrist_up, wrist_down (similar)
   → 6 CSV files

 train_base_model.py
   ↓ Load all 6 CSV files = 630 total repetitions
   → 630 × 600 = 378,000 EMG samples total
   ↓ Extract features, normalize, split 60/15/25
   ↓ Train CNN+LSTM for 100 epochs
   → Model converges to 95% test accuracy
   → Save base_model_best.pth

 calibrate_model.py (Session 1)
   ↓ Load base model
   ↓ Collect 6 quick cal samples (3s each) = 6 × 600 = 3,600 samples
   ↓ Freeze CNN+LSTM, fine-tune FC layers (20 epochs)
   → Save session_1_adapted_model.pth

Day 2 - Band Repositioned:
──────────────────────────
 calibrate_model.py (Session 2)
   ↓ Load base model (again)
   ↓ Collect 6 quick cal samples = 3,600 samples (just 30 seconds!)
   ↓ Freeze CNN+LSTM, fine-tune FC layers (20 epochs)
   → Save session_2_adapted_model.pth

 predict_realtime.py
   ↓ Load session_2_adapted_model.pth
   ↓ Stream EMG @ 200 Hz
   ↓ Sliding windows: 0.5s window, 0.25s stride
   → Predictions every 0.25 seconds
   → Output: "REST", "FIST", "WRIST_UP", etc. with confidence


CSV FORMAT DETAIL
═════════════════

rest.csv (example first 10 rows + random middle rows):
────────────────────────────────────────────────────
ms,adc_ch1,volt_ch1,adc_ch2,volt_ch2,rep
0,512,0.412,478,0.385,1
5,514,0.414,480,0.387,1
10,516,0.415,482,0.388,1
15,518,0.416,484,0.390,1
20,520,0.418,486,0.392,1
25,522,0.420,488,0.393,1
30,524,0.422,490,0.395,1
35,526,0.424,492,0.396,1
40,528,0.425,494,0.398,1
45,530,0.427,496,0.400,1
...
2950,498,0.401,465,0.375,10
2955,496,0.400,463,0.373,10
2960,494,0.398,461,0.371,10
...
5990,520,0.418,486,0.392,15
5995,522,0.420,488,0.393,15

Rows: 600 samples/rep × 15 reps × 105 session-reps = 63,000 rows
Columns: 6 (ms, adc_ch1, volt_ch1, adc_ch2, volt_ch2, rep)


MODEL ARCHITECTURE SUMMARY
═══════════════════════════

Input Shape: (Batch=32, Channels=2, Samples=600)
                        ↓
Conv1D(in=2, out=32, kernel=5) + BN + ReLU + MaxPool2
                    Output: (32, 300)
                        ↓
Conv1D(in=32, out=64, kernel=5) + BN + ReLU + MaxPool2
                    Output: (64, 150)
                        ↓
Conv1D(in=64, out=128, kernel=5) + BN + ReLU + MaxPool2
                    Output: (128, 75)
                        ↓
Reshape for LSTM: (128, 75) → Transpose → (75, 128)
                        ↓
LSTM(input=128, hidden=64, num_layers=2, dropout=0.3)
                    Output: (75, 64)
                        ↓
Take last state: (64,)
                        ↓
FC(64 → 32) + ReLU + Dropout(0.3)
                    Output: (32,)
                        ↓
FC(32 → 6) [6 gestures]
                    Output: (6,)
                        ↓
Softmax → Probabilities per gesture
                    Output: (6,) normalized

Total Parameters: ~155K (mostly CNN+LSTM)
Trainable during calibration: ~2K (only FC layers)
Frozen during calibration: ~153K (CNN+LSTM)
═════════════════════════════════════════════════════════════════════════════
```

## Key Insight: Why Fast Calibration Works

```
Traditional Fine-Tuning (❌ Still slow):
  Retrain entire model on new data
  All layers adjust
  Takes hours or days
  
Transfer Learning + Layer Freezing (✓ Fast!):
  
  Feature Extractors (CNN+LSTM):
  ┌─────────────────────────────────────────┐
  │ These learned DEEP patterns about EMG   │
  │ - How raw signals correlate with muscle │
  │ - Gesture temporal dynamics             │
  │ - Frequency patterns                    │
  │                                         │
  │ These DON'T change with electrode shift!│
  │ ✓ FREEZE THESE                          │
  └─────────────────────────────────────────┘
  
  Classification Head (FC layers):
  ┌─────────────────────────────────────────┐
  │ This maps features → gesture class      │
  │                                         │
  │ These CHANGE with electrode position!   │
  │ → New electrode = different signal gain │
  │ → Different baseline voltage            │
  │ → Need to recalibrate output mapping    │
  │                                         │
  │ 🔥 FINE-TUNE THIS (only 2K params!)    │
  └─────────────────────────────────────────┘
  
Result:
  - 99% parameters frozen (stable)
  - 1% parameters trained (adaptive)
  - Takes 30 seconds vs 10+ minutes
  - Handles electrode shift perfectly
```

---

**This architecture enables the "write once, calibrate daily" workflow!** 🎯
