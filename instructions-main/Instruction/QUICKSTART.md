# Quick Start Guide

## TL;DR - 4 Steps to Working Gesture Recognition

### Step 1: Collect Data (One-Time Setup - ~42 minutes)
```bash
python azmuth.py
```
- Performs 6 gestures × 7 sessions
- Each session: 45 seconds (15 × 3-second reps)
- Saves to `data_raw/` as CSV files
- **At 200Hz sampling rate** ✓

### Step 2: Train Model (One-Time - ~10 minutes)
```bash
python train_base_model.py
```
- Creates CNN+LSTM model
- Trains on all collected data
- Saves to `models/base_model_best.pth`
- **Expected accuracy: 90-98%** (this is good for intra-session!)

### Step 3: Quick Calibration (Every Session - ~30 seconds)
```bash
python calibrate_model.py
```
- Quick gesture recording (3-5s per gesture)
- Adapts model to new electrode position
- Takes only 20-30 seconds!
- Solves inter-session variability

### Step 4: Real-Time Prediction (Anytime)
```bash
python predict_realtime.py
```
- Streams EMG data from sensor
- Predicts gestures in real-time
- Run anytime after calibration

## What's Different from Your Old Script?

| Feature | Old (azmuth.py) | New System |
|---------|-----------------|-----------|
| Sampling Rate | 5 Hz | **200 Hz** ✓ |
| Sessions | Single | 7 sessions |
| Adaptation | None | Fast fine-tuning ✓ |
| Inter-session Variability | ❌ Fails | ✓ Handled |
| Setup Time | N/A | ~42 min data + 10 min training |
| Recalibration Time | N/A | **~30 seconds** ✓ |

## Key Innovation: Model Adaptation

**Problem**: When you remove the EMG band and put it back on, the model fails due to electrode shift

**Solution**: Instead of collecting and training 42 more minutes of data:
1. Record 3-5 seconds per gesture
2. Fine-tune ONLY the last layer of the model
3. Ready in 30 seconds!

This is **much faster** than retraining from scratch.

## Sampling Rate Update

Your comment said current script collects at 5Hz, but you need 200Hz.

**Changes made to azmuth.py:**
```python
# OLD:
# (collected at whatever rate serial provided)

# NEW:
SAMPLING_RATE = 200  # Hz - explicit configuration
SAMPLES_PER_REP = int(SAMPLING_RATE * DURATION_PER_REP)  # 600 samples
EXPECTED_SAMPLES_TOTAL = int(SAMPLING_RATE * TOTAL_DURATION)  # 9000 samples

# Console output now shows:
# ✅ Saving data to: data_raw/rest.csv (1827 rows)
#    Sampling rate achieved: 200.3 Hz  ← Verification!
```

**To achieve 200Hz at the hardware level**, your ESP32 firmware must:
```cpp
// Arduino/ESP32 code (not in this Python repo, but you need this)
#define SAMPLING_RATE 200  // Hz
#define SAMPLE_INTERVAL 1000000 / SAMPLING_RATE  // microseconds

void setup() {
    Serial.begin(115200);
    digitalWrite(26, HIGH);  // Enable ADC
}

void loop() {
    // Read at 200Hz interval
    unsigned long start = micros();
    
    int adc1 = analogRead(34);  // Pin 34
    int adc2 = analogRead(35);  // Pin 35
    
    Serial.print(adc1);
    Serial.print(",");
    Serial.println(adc2);
    
    // Wait for next sample interval
    while (micros() - start < SAMPLE_INTERVAL) {}
}
```

If your ESP32 isn't configured for 200Hz, the Python script will still collect data, but it'll be at whatever rate your ESP32 sends it. The Python script will report the actual achieved rate.

## Expected Results

### After Data Collection
```
data_raw/
├── rest.csv          (105 repetitions)
├── open.csv          (105 repetitions)
├── fist.csv          (105 repetitions)
├── strong_fist.csv   (105 repetitions)
├── wrist_up.csv      (105 repetitions)
└── wrist_down.csv    (105 repetitions)
```

Each CSV: ~600 samples per rep × 105 reps = ~63,000 rows

### After Model Training
```
Models Trained:
✅ Test Accuracy: 95.2%
```

Graphs saved:
- `training_history.png` - Loss and accuracy curves

### After Calibration
```
Adapting model to user...
   Epoch [5/20] Loss: 0.2103 | Acc: 92.34%
   Epoch [10/20] Loss: 0.1256 | Acc: 95.61%
   Epoch [15/20] Loss: 0.0892 | Acc: 97.23%
   Epoch [20/20] Loss: 0.0654 | Acc: 98.11%

✅ Adaptation complete! (Calibration accuracy: 98.11%)
   Run: python predict_realtime.py
```

### During Real-Time Prediction
```
[0.5s] REST            (87.3%)
[1.0s] FIST            (92.1%)
[1.5s] FIST            (94.6%)
[2.0s] OPEN            (84.6%)
[2.5s] REST            (88.2%)
[3.0s] WRIST_UP        (91.2%)
[3.5s] WRIST_UP        (93.7%)
[4.0s] REST            (86.1%)
```

## Requirements File

All dependencies are in `requirements.txt`:
```
numpy
scipy
matplotlib
pandas
scikit-learn
torch
torchvision
pyserial
tqdm
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

## File Summary

| File | Purpose | Key Change |
|------|---------|-----------|
| `azmuth.py` | Collect EMG data | **200Hz sampling** ✓ |
| `train_base_model.py` | Train CNN+LSTM | NEW - full training pipeline |
| `calibrate_model.py` | Fast adaptation | NEW - 30s recalibration |
| `predict_realtime.py` | Real-time inference | NEW - live predictions |

## Typical Session

**Day 1 (Setup - 1 hour total)**
1. Run `azmuth.py` - collect 7 sessions (42 min)
2. Run `train_base_model.py` - train model (10 min)
3. Run `calibrate_model.py` - calibrate once (1 min)
4. Run `predict_realtime.py` - demo! (1 min)

**Day 2+ (Quick Calibration - 1 minute total)**
1. Put on band
2. Run `calibrate_model.py` - recalibrate (30 sec)
3. Run `predict_realtime.py` - ready to go!

## Common Questions

**Q: Why does the base model have 90%+ accuracy?**
A: Because all data is from the same user on the same day. Real challenge is when electrode shifts (inter-session). That's why calibration is needed.

**Q: What if I want to share this with another user?**
A: You need to collect data from them too. Multi-user models are more complex (and better). Current system is single-user, personalized.

**Q: Can I use this without retraining?**
A: No - you need Step 1-2 once per user. But Step 3 (calibration) only takes 30 seconds each day.

**Q: My accuracy drops after a day - what's wrong?**
A: Electrode shift! Run calibration again (`calibrate_model.py`). 30 seconds fixes it.

**Q: How many samples do I need?**
A: Current setup: 7 sessions × 15 reps × 6 gestures = 630 total samples. Minimum: ~200-300.

---

**Ready to go? Start with Step 1!** 🎯

```bash
python azmuth.py
```
