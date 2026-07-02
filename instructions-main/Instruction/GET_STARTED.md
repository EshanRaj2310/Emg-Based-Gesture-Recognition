# Step-by-Step Instructions to Get Your System Working

## Prerequisites
- ESP32 microcontroller with 2 EMG sensors connected to pins 34 & 35
- USB cable to connect ESP32 to your computer
- Python 3.7+ installed on your system

---

## Phase 1: Setup (Do This First)

### Step 1.1: Install Python Dependencies
```bash
cd c:\Users\bharg\Downloads\Omnitrix-main\Omnitrix-main
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed numpy scipy matplotlib pandas scikit-learn torch ...
```

### Step 1.2: Verify ESP32 Connection
1. Connect ESP32 to your computer via USB cable
2. Determine which COM port it uses:
   ```bash
   # In PowerShell, list serial ports
   [System.IO.Ports.SerialPort]::GetPortNames()
   ```
   
3. If it shows `COM3`, `COM5`, or similar - you're good!
4. Update the PORT in scripts if needed:
   - Open `azmuth.py`, find `PORT = "COM5"` (line ~31)
   - Change to your port (e.g., `PORT = "COM3"`)

### Step 1.3: Verify ESP32 Firmware
Your ESP32 must be programmed to read EMG at 200Hz. Upload this code to ESP32:

```cpp
// Arduino IDE - select "Board: ESP32 Dev Module"
#define SAMPLING_RATE 200  // Hz
#define SAMPLE_INTERVAL 1000000 / SAMPLING_RATE  // microseconds

unsigned long lastSample = 0;

void setup() {
    Serial.begin(115200);
    pinMode(34, INPUT);  // Channel 1
    pinMode(35, INPUT);  // Channel 2
}

void loop() {
    unsigned long now = micros();
    
    if (now - lastSample >= SAMPLE_INTERVAL) {
        int adc1 = analogRead(34);
        int adc2 = analogRead(35);
        
        Serial.print(adc1);
        Serial.print(",");
        Serial.println(adc2);
        
        lastSample = now;
    }
}
```

### Step 1.4: Test Serial Connection
```bash
python -c "
import serial
import time

ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(1)
print('Reading 10 samples from ESP32:')
for i in range(10):
    if ser.in_waiting:
        line = ser.readline().decode().strip()
        print(f'{i}: {line}')
ser.close()
"
```

**Expected output:**
```
Reading 10 samples from ESP32:
0: 512,478
1: 514,480
2: 516,482
...
```

If you see data → **Continue to Phase 2**
If no data → Check COM port, ESP32 USB cable, and Arduino code

---

## Phase 2: Data Collection (42 minutes)

### Step 2.1: Prepare Your Setup
1. **EMG Band Placement:**
   - Forearm, about 4-5cm below elbow
   - Both sensors in same orientation
   - Use electrode gel for good contact

2. **Comfortable Position:**
   - Sit in a chair with forearm supported
   - Relax between repetitions

### Step 2.2: Run Data Collection
```bash
python azmuth.py
```

**What happens:**
```
🔌 Connecting to ESP32...
✅ Connected to COM5 at 115200 baud.

=== Gesture: REST ===
Meaning: Neutral / no message
📊 Expected data points: ~9000 samples at 200Hz

👉 Prepare for 'rest'. Press Enter to start...
[Press Enter]

Starting continuous recording...
Recording repetition 1 of 15 (3s)...
Recording repetition 2 of 15 (3s)...
...
Recording repetition 15 of 15 (3s)...

Done. Continuous recording finished.
✅ Saving data to: data_raw/rest.csv (9000 rows)
   Sampling rate achieved: 200.1 Hz
📊 Saved plot to: graphs/rest.png

=== Gesture: OPEN ===
[continues for all 6 gestures]
```

### Step 2.3: What to Do for Each Gesture

**REST:**
- Just relax your arm, no muscle contraction

**OPEN:**
- Spread fingers wide like stopping a car
- Hold steady for 1.5 seconds, release for 1.5 seconds

**FIST:**
- Light squeeze, fingers curled
- Hold steady for 1.5 seconds, release for 1.5 seconds

**STRONG_FIST:**
- Maximum squeeze, tense the entire forearm
- Hold steady for 1.5 seconds, release for 1.5 seconds

**WRIST_UP:**
- Bend wrist upward (towards you)
- Hold steady for 1.5 seconds, release for 1.5 seconds

**WRIST_DOWN:**
- Bend wrist downward (away from you)
- Hold steady for 1.5 seconds, release for 1.5 seconds

### Step 2.4: Verify Collection Success
```bash
# Check files were created
dir data_raw\

# Should show 6 CSV files:
# fist.csv, open.csv, rest.csv, strong_fist.csv, wrist_down.csv, wrist_up.csv

# Check one file
python -c "
import pandas as pd
df = pd.read_csv('data_raw/rest.csv')
print(f'Rows: {len(df)} (expected ~9000)')
print(f'Columns: {list(df.columns)}')
print(f'Sample rate: {len(df) / 45:.1f} Hz (expected ~200 Hz)')
"
```

**Expected output:**
```
Rows: 9000 (expected ~9000)
Columns: ['ms', 'adc_ch1', 'volt_ch1', 'adc_ch2', 'volt_ch2', 'rep']
Sample rate: 200.0 Hz (expected ~200 Hz)
```

---

## Phase 3: Train Base Model (10 minutes)

### Step 3.1: Run Training
```bash
python train_base_model.py
```

**What happens:**
```
============================================================
EMG GESTURE RECOGNITION - BASE MODEL TRAINING
============================================================

📂 Loading EMG data from CSV files...
✅ rest: 105 repetitions loaded
✅ open: 105 repetitions loaded
✅ fist: 105 repetitions loaded
✅ strong_fist: 105 repetitions loaded
✅ wrist_up: 105 repetitions loaded
✅ wrist_down: 105 repetitions loaded

✅ Total data: 630 repetitions across 6 gestures

📊 Data split:
   Train: 378 samples
   Val:   94 samples
   Test:  158 samples

🧠 Creating model...
   Model parameters: 155,437

🚀 Training for 100 epochs...
Epoch [10/100] | Train Loss: 0.8234 | Val Loss: 0.5612 | Val Acc: 87.23%
Epoch [20/100] | Train Loss: 0.3456 | Val Loss: 0.2189 | Val Acc: 93.62%
...
Epoch [100/100] | Train Loss: 0.0123 | Val Loss: 0.0456 | Val Acc: 97.87%

🧪 Testing on held-out test set...
✅ Test Accuracy: 95.57%

💾 Model saved to: models/base_model_best.pth
📊 Training history saved to: training_history.png

============================================================
✅ Base model training complete!
============================================================
```

### Step 3.2: Verify Training Success
```bash
# Check model files
dir models\

# Should show:
# base_model_best.pth
# base_model_final.pth
# gesture_mappings.pkl
# training_history.png
```

**Expected accuracy: 90-98%** (This is good! Means model learned well)

---

## Phase 4: Quick Calibration (30 seconds)

### Step 4.1: Prepare for Calibration
The calibration adapts the model to YOUR specific electrode position and skin.

```bash
python calibrate_model.py
```

**What happens:**
```
============================================================
CALIBRATION MODE - Quick gesture recording
============================================================
📊 Quick calibration: 3s per gesture
   Total time: ~18s

✅ Connected to COM5 at 115200 baud

=== Collecting calibration data for 'rest'...
   Duration: 3 seconds
👉 Prepare gesture. Press Enter to start...
[Press Enter]
✅ Collected 600 samples (3.00s actual)

=== Collecting calibration data for 'open'...
[continues for all 6 gestures]

============================================================
✅ Calibration data collection complete!
============================================================

✅ Calibration data preprocessed: 6 samples

🧊 Freezing feature extraction layers (CNN + LSTM)
🔥 Training only: FC layers (2,066 params)

🚀 Adapting model to user...
   Epoch [5/20] Loss: 0.2103 | Acc: 92.34%
   Epoch [10/20] Loss: 0.1256 | Acc: 95.61%
   Epoch [15/20] Loss: 0.0892 | Acc: 97.23%
   Epoch [20/20] Loss: 0.0654 | Acc: 98.11%

✅ Adaptation complete! (Calibration accuracy: 98.11%)
💾 Adapted model saved to: models/adapted_models/session_1_adapted_model.pth

============================================================
✅ Ready for real-time prediction!
   Run: python predict_realtime.py
============================================================
```

### Step 4.2: Perform Calibration Gestures
Same as Phase 2 - just hold each gesture for 3 seconds:
- REST: Relax
- OPEN: Spread fingers
- FIST: Light squeeze
- STRONG_FIST: Maximum squeeze
- WRIST_UP: Bend wrist up
- WRIST_DOWN: Bend wrist down

---

## Phase 5: Real-Time Prediction (Live)

### Step 5.1: Start Real-Time Prediction
```bash
python predict_realtime.py
```

**What you see:**
```
============================================================
EMG GESTURE RECOGNITION - REAL-TIME PREDICTION
============================================================
Window size: 100 samples (0.50s)
Stride: 50 samples (0.25s)
Press Ctrl+C to stop

[0.5s] REST            (87.3%)
[1.0s] FIST            (92.1%)
[1.5s] FIST            (94.6%)
[2.0s] OPEN            (84.6%)
[2.5s] REST            (88.2%)
[3.0s] WRIST_UP        (91.2%)
[3.5s] WRIST_UP        (93.7%)
[4.0s] REST            (86.1%)
```

### Step 5.2: Test Each Gesture
Perform gestures and watch predictions:
- **REST:** Should show "REST" with high confidence
- **FIST:** Should show "FIST" or "STRONG_FIST"
- **OPEN:** Should show "OPEN"
- **WRIST_UP:** Should show "WRIST_UP"
- **WRIST_DOWN:** Should show "WRIST_DOWN"

Press **Ctrl+C** to stop

---

## Complete Timeline

| Phase | Action | Time | Output |
|-------|--------|------|--------|
| 1 | Setup & verify | 10 min | Serial data flowing |
| 2 | Collect data | 42 min | 6 CSV files |
| 3 | Train model | 10 min | base_model_best.pth |
| 4 | Calibrate | 30 sec | session_1_adapted_model.pth |
| 5 | Predict | Live | Real-time gestures |

**Total initial setup: ~1 hour**

---

## Daily Use After Setup

### Every Day (Band On for First Time)
```bash
# Quick 30-second calibration
python calibrate_model.py

# Then predict
python predict_realtime.py
```

### If You Remove & Replace Band
```bash
# Just calibrate again (30 seconds)
python calibrate_model.py

# Predictions will work again!
```

### To Collect More Data (Optional)
```bash
# Collect additional sessions to improve base model
python azmuth.py

# Retrain base model
python train_base_model.py

# Recalibrate
python calibrate_model.py
```

---

## Troubleshooting Quick Links

**Problem → Solution:**
- No serial data → Check COM port in `azmuth.py` (line ~31)
- Low sampling rate → Verify ESP32 firmware is uploading correctly
- Poor accuracy → Check signal quality in `graphs/*.png` plots
- Model fails after day → Run `calibrate_model.py` again (30 seconds)

See **`TROUBLESHOOTING.md`** for detailed help on any issues.

---

## Key Files Reference

| File | Purpose | When to Run |
|------|---------|------------|
| `azmuth.py` | Collect EMG data | Phase 2 (once per dataset) |
| `train_base_model.py` | Train model | Phase 3 (once per dataset) |
| `calibrate_model.py` | Quick adaptation | Phase 4 (daily when band on) |
| `predict_realtime.py` | Live prediction | Phase 5 (anytime after calib) |

---

## Success Criteria

✓ Can you run `python azmuth.py` and collect data?
✓ Do you see CSV files in `data_raw/` with ~9000 rows each?
✓ Can you run `python train_base_model.py` and get >85% test accuracy?
✓ Can you run `python calibrate_model.py` without errors?
✓ Can you run `python predict_realtime.py` and see gesture predictions?

If all ✓, **you're done! System is working!**

---

## Next Steps

1. **Follow Phases 1-5 in order** (today, ~1 hour total)
2. **Practice with Phase 5** until comfortable with predictions
3. **If accuracy drops** → Run calibration again
4. **If you want better accuracy** → Collect more data in Phase 2, retrain in Phase 3

---

**Start with:** `pip install -r requirements.txt`

Good luck! 🎯
