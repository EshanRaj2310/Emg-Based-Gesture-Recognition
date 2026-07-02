# Quick Checklist - Follow This Order

## ✅ PRE-SETUP (5 minutes)

- [ ] ESP32 connected to computer via USB
- [ ] Electrode sensors attached to ESP32 pins 34 & 35
- [ ] EMG band placed on forearm
- [ ] Python 3.7+ installed
- [ ] In correct directory: `c:\Users\bharg\Downloads\Omnitrix-main\Omnitrix-main`

---

## ✅ STEP 1: Install & Verify (10 minutes)

```bash
# Run this command
pip install -r requirements.txt

# Wait for: "Successfully installed..."
```

**Then check serial connection:**
```bash
# Find your COM port
python -c "import serial.tools.list_ports; [print(p.device) for p in serial.tools.list_ports.comports()]"

# Result should show something like: COM3, COM5, etc.
```

**Update azmuth.py if needed:**
- Open `azmuth.py` in editor
- Line ~31: `PORT = "COM5"` 
- Change COM5 to your actual port (if different)

- [ ] Dependencies installed
- [ ] Found COM port
- [ ] Updated PORT in scripts (if needed)

---

## ✅ STEP 2: Collect EMG Data (42 minutes)

```bash
python azmuth.py
```

**For each of 6 gestures, you'll be prompted:**

1. **REST**
   - [ ] Press Enter
   - [ ] Relax your arm (no contraction)
   - [ ] Wait 45 seconds
   - [ ] See: ✅ Saving data to: data_raw/rest.csv

2. **OPEN**
   - [ ] Press Enter
   - [ ] Spread fingers wide × 15 times (3s hold, 1.5s off)
   - [ ] Wait 45 seconds total
   - [ ] See: ✅ Saving data to: data_raw/open.csv

3. **FIST**
   - [ ] Press Enter
   - [ ] Light squeeze × 15 times (3s hold, 1.5s off)
   - [ ] Wait 45 seconds total
   - [ ] See: ✅ Saving data to: data_raw/fist.csv

4. **STRONG_FIST**
   - [ ] Press Enter
   - [ ] Maximum squeeze × 15 times (3s hold, 1.5s off)
   - [ ] Wait 45 seconds total
   - [ ] See: ✅ Saving data to: data_raw/strong_fist.csv

5. **WRIST_UP**
   - [ ] Press Enter
   - [ ] Bend wrist upward × 15 times (3s hold, 1.5s off)
   - [ ] Wait 45 seconds total
   - [ ] See: ✅ Saving data to: data_raw/wrist_up.csv

6. **WRIST_DOWN**
   - [ ] Press Enter
   - [ ] Bend wrist downward × 15 times (3s hold, 1.5s off)
   - [ ] Wait 45 seconds total
   - [ ] See: ✅ Saving data to: data_raw/wrist_down.csv

**After all 6 gestures:**
```bash
# Verify files were created
dir data_raw\

# Should show 6 CSV files with ~9000 rows each
```

- [ ] Collected all 6 gestures
- [ ] See CSV files in `data_raw/`
- [ ] Plots saved in `graphs/`

---

## ✅ STEP 3: Train Model (10 minutes)

```bash
python train_base_model.py
```

**Watch the training:**
```
🚀 Training for 100 epochs...
Epoch [10/100] | Train Loss: 0.8234 | Val Loss: 0.5612 | Val Acc: 87.23%
Epoch [20/100] | Train Loss: 0.3456 | Val Loss: 0.2189 | Val Acc: 93.62%
... (many epochs) ...
Epoch [100/100] | ...

🧪 Testing on held-out test set...
✅ Test Accuracy: 95.57%
```

**Expected:** Test accuracy 90-98%

- [ ] Training completed
- [ ] Test accuracy > 85%
- [ ] See `models/base_model_best.pth`

---

## ✅ STEP 4: Quick Calibration (30 seconds)

```bash
python calibrate_model.py
```

**For each gesture, press Enter and hold for 3 seconds:**

1. [ ] REST - Press Enter, relax (3s)
2. [ ] OPEN - Press Enter, spread fingers (3s)
3. [ ] FIST - Press Enter, light squeeze (3s)
4. [ ] STRONG_FIST - Press Enter, maximum squeeze (3s)
5. [ ] WRIST_UP - Press Enter, bend wrist up (3s)
6. [ ] WRIST_DOWN - Press Enter, bend wrist down (3s)

**Watch adaptation:**
```
🧊 Freezing feature extraction layers (CNN + LSTM)
🔥 Training only: FC layers (2,066 params)

🚀 Adapting model to user...
   Epoch [5/20] Loss: 0.2103 | Acc: 92.34%
   ...
   Epoch [20/20] Loss: 0.0654 | Acc: 98.11%

✅ Adaptation complete! (Calibration accuracy: 98.11%)
```

- [ ] Calibration completed
- [ ] See `models/adapted_models/session_1_adapted_model.pth`

---

## ✅ STEP 5: Real-Time Prediction (Live!)

```bash
python predict_realtime.py
```

**You should see:**
```
============================================================
🎯 REAL-TIME GESTURE PREDICTION
============================================================
Press Ctrl+C to stop

[0.5s] REST            (87.3%)
[1.0s] FIST            (92.1%)
[1.5s] FIST            (94.6%)
[2.0s] OPEN            (84.6%)
...
```

**Test each gesture:**
- [ ] REST - Relax, see "REST" with high confidence
- [ ] FIST - Squeeze, see "FIST" with high confidence
- [ ] OPEN - Spread fingers, see "OPEN" with high confidence
- [ ] STRONG_FIST - Hard squeeze, see "STRONG_FIST" with high confidence
- [ ] WRIST_UP - Bend up, see "WRIST_UP" with high confidence
- [ ] WRIST_DOWN - Bend down, see "WRIST_DOWN" with high confidence

**Press Ctrl+C to stop**

- [ ] All predictions working
- [ ] Accuracy acceptable

---

## 🎉 SUCCESS!

If you checked all boxes → **Your system is working!**

---

## 📋 Daily Usage Going Forward

### First time band is put on (every day):
```bash
python calibrate_model.py    # 30 seconds
python predict_realtime.py   # Live predictions
```

### If predictions drop in accuracy:
```bash
python calibrate_model.py    # Quick fix (30 seconds)
```

### If you want better accuracy overall:
```bash
python azmuth.py             # Collect more data
python train_base_model.py   # Retrain
python calibrate_model.py    # Recalibrate
```

---

## 🆘 Issues?

**Can't run pip install?**
→ See `TROUBLESHOOTING.md` section 9

**No serial data?**
→ See `TROUBLESHOOTING.md` section 1

**Low accuracy?**
→ See `TROUBLESHOOTING.md` section 2

**Any other issue?**
→ Read `TROUBLESHOOTING.md`

---

**Total Time: ~1 hour initial setup**

**Then: 30 seconds daily calibration for perfect predictions**

**Start now:** `pip install -r requirements.txt`

🎯 Good luck!
