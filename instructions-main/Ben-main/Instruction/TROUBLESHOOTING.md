# Troubleshooting Guide

## Common Issues & Solutions

### 1. Serial Connection Issues

#### Problem: `Serial connection failed: Serial port COM5 not found`

**Solutions:**
```python
# Check available serial ports
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"{port.device}: {port.description}")

# Output might show:
# COM3: Silicon Labs CP210x USB to UART Bridge
# COM5: USB-SERIAL CH340
```

**Fix:**
```python
# Update PORT in your scripts
PORT = "COM3"  # Change to correct port
# or
PORT = "COM5"
```

#### Problem: `Serial permission denied`

**Windows:**
```
This is usually not an issue on Windows
Re-install CH340 or CP210x drivers from manufacturer
```

**Linux:**
```bash
sudo usermod -a -G dialout $USER
# Then logout and login
```

#### Problem: `No data received from serial`

**Checklist:**
- [ ] ESP32 connected via USB cable
- [ ] Cable has data pins (not just power)
- [ ] ESP32 firmware is running
- [ ] Arduino IDE Serial Monitor shows data at same baud rate
- [ ] Baud rate matches: 115200

**Test:**
```python
import serial
ser = serial.Serial("COM5", 115200, timeout=1)
ser.reset_input_buffer()

# Read 10 lines
for i in range(10):
    if ser.in_waiting:
        line = ser.readline().decode().strip()
        print(f"Line {i}: {line}")
    else:
        print("No data available")
```

---

### 2. Data Collection Issues

#### Problem: `Low sampling rate achieved: 15.2 Hz instead of 200 Hz`

**Cause:** ESP32 firmware not sending data fast enough

**Solutions:**

1. **Verify ESP32 is configured for 200Hz:**
```cpp
// Arduino/ESP32 code
#define SAMPLING_RATE 200  // Hz

unsigned long lastSample = 0;
unsigned long sampleInterval = 1000000 / SAMPLING_RATE;  // microseconds

void loop() {
    unsigned long now = micros();
    if (now - lastSample >= sampleInterval) {
        int adc1 = analogRead(34);
        int adc2 = analogRead(35);
        Serial.print(adc1);
        Serial.print(",");
        Serial.println(adc2);
        
        lastSample = now;
    }
}
```

2. **Check baud rate - 115200 should be fine for 200Hz**
```
At 200 samples/second × 2 channels × ~10 bytes per sample:
= 200 × 2 × 10 = 4000 bytes/sec
= 32,000 bits/sec << 115,200 baud ✓
```

3. **Disable WiFi/Bluetooth on ESP32** (they might interfere)
```cpp
WiFi.mode(WIFI_OFF);
btStop();
```

#### Problem: `Noisy or erratic ADC values`

**Causes:**
- Poor electrode contact with skin
- Electrode paste dried out
- EMG electrodes not properly shielded
- Sampling too fast relative to ADC settling time

**Solutions:**
1. Clean electrode area and reapply gel
2. Check electrode cable connections
3. Add 100µF capacitor across ADC input
4. Increase ADC sampling time in ESP32 configuration
```cpp
analogSetAttenuation(ADC_11db);  // Reduces noise
analogSetClockDiv(255);  // Slower ADC sampling
```

---

### 3. Model Training Issues

#### Problem: `FileNotFoundError: data_raw/rest.csv not found`

**Cause:** Data collection script didn't run or didn't save files

**Solutions:**
```bash
# Check if data_raw directory exists
ls data_raw/
# If not, run collection:
python azmuth.py

# Verify CSV files
dir data_raw\
# Should show: rest.csv, open.csv, fist.csv, etc.
```

#### Problem: `Model accuracy very low: <60%`

**Possible Causes:**

1. **Poor signal quality**
   - Check PNG graphs in `graphs/` directory
   - Look for noise vs. clear gesture patterns
   - Re-collect data with better electrode contact

2. **Inconsistent gesture performance**
   - Perform gestures the same way every time
   - Same muscle groups, same contraction level
   - Different approaches produce different signals

3. **Too few training samples**
   - Current setup: 630 total samples
   - Try collecting 10-15 sessions per gesture instead of 7
   - More data = better model

4. **Misaligned repetitions**
   - Verify `rep` column in CSV is correct
   - Ensure each "rep" is exactly 3 seconds
   - Check plot shows 15 distinct peaks (one per rep)

**Debug:**
```python
import pandas as pd

# Examine one CSV
df = pd.read_csv("data_raw/rest.csv")
print(df.head(20))
print(f"Total rows: {len(df)}")
print(f"Unique reps: {df['rep'].unique()}")
print(f"Expected: ~600 samples × 15 reps = ~9000 rows")

# Check signal quality
import matplotlib.pyplot as plt
ch1 = df['adc_ch1'].values
plt.plot(ch1)
plt.title("Channel 1 Signal")
plt.show()
```

#### Problem: `Batch size error or OOM (Out of Memory)`

**Solutions:**
```python
# In train_base_model.py
BATCH_SIZE = 16  # Reduce from 32
# or on very limited systems:
BATCH_SIZE = 8

# Also reduce data loading:
num_workers = 0  # Don't use multiprocessing
```

---

### 4. Calibration Issues

#### Problem: `Calibration accuracy low: <70%`

**Causes:**
1. Poor quality calibration data (noisy electrodes)
2. Inconsistent gesture performance during calibration
3. Misaligned electrode position from training data
4. Base model not well-trained

**Solutions:**
```
1. Ensure base model accuracy >90%
2. Clean electrodes thoroughly
3. Perform calibration gestures clearly and consistently
4. Collect 5-10 seconds per gesture instead of 3
5. Do multiple calibration sessions and average
```

#### Problem: `ModuleNotFoundError: No module named 'torch'`

**Fix:**
```bash
pip install torch torchvision  # CPU version
# or with GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Problem: `Adapted model not found after calibration`

**Check:**
```python
import os

# Verify adapted_models directory exists
adapted_dir = "models/adapted_models"
if os.path.exists(adapted_dir):
    print("✓ Directory exists")
    files = os.listdir(adapted_dir)
    print(f"Files: {files}")
else:
    print("✗ Directory missing - run calibrate_model.py again")
```

---

### 5. Real-Time Prediction Issues

#### Problem: `Model not found: models/adapted_models/session_1_adapted_model.pth`

**Cause:** Calibration script didn't run or didn't save model

**Fix:**
```bash
python calibrate_model.py
# Wait for "✅ Adaptation complete!"
# Then:
python predict_realtime.py
```

#### Problem: `Predictions all showing "waiting"`

**Cause:** Buffer not filling (no serial data or parsing errors)

**Debug:**
```python
# Test serial connection independently
import serial
ser = serial.Serial("COM5", 115200, timeout=1)

print("Reading 20 serial lines...")
for i in range(20):
    if ser.in_waiting:
        line = ser.readline().decode('errors=ignore').strip()
        print(f"{i}: {line}")
        
        # Try to parse
        import re
        nums = re.findall(r"\d+", line)
        if len(nums) >= 2:
            print(f"   Parsed: {nums[0]}, {nums[1]} ✓")
        else:
            print(f"   Parse failed ✗")
```

#### Problem: `Accuracy drops after a few minutes of prediction`

**Cause:** Model drift due to electrode slippage

**Solution:**
```
This is normal! Electrode position shifts slightly over time.
Re-run calibration:
python calibrate_model.py
```

#### Problem: `Predictions lag/stutter`

**Cause:** 
- GPU memory issues
- CPU overload
- Serial buffer overflow

**Solutions:**
```python
# In predict_realtime.py
WINDOW_SIZE = 200  # Reduce from 100 for faster predictions
WINDOW_STRIDE = 100  # Larger stride = fewer predictions/sec

# Or on CPU:
# GPU inference should be smooth, CPU might be slow
```

---

### 6. Performance Issues

#### Problem: `Training is very slow (>30 min per epoch)`

**Cause:** 
- Running on CPU (expected to be slow)
- Data loading bottleneck
- Too large batch size

**Solutions:**
```python
# Use GPU if available
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Or reduce batch size
BATCH_SIZE = 16  # from 32

# Or reduce dataset
# Collect fewer sessions (5 instead of 7)
```

#### Problem: `Real-time predictions have high latency (>1 second)`

**Cause:** Window size too large

**Fix:**
```python
# In predict_realtime.py
WINDOW_SIZE = 50  # Reduce from 100 (0.25s instead of 0.5s)
WINDOW_STRIDE = 25  # Update stride accordingly
```

---

### 7. Accuracy & Reliability

#### Problem: `Model works great day 1, fails day 2`

**Cause:** Inter-session variability (electrode shift)

**Solution:** ✓ This is WHY we have calibration!
```bash
# Day 2: Just recalibrate (30 seconds)
python calibrate_model.py
# Wait for "✅ Adaptation complete!"
# Model now works great again!
```

#### Problem: `Specific gesture always misclassified (e.g., "fist" → "strong_fist")`

**Cause:** 
- Similar muscle activation patterns
- Inconsistent gesture performance
- Model confusion

**Solutions:**
1. **Exaggerate difference during collection:**
   - Light fist: minimal muscle contraction
   - Strong fist: maximum contraction (tense, squeeze hard)

2. **Collect more data** for confusing gesture pairs

3. **Visualize confusion:**
```python
# After training, examine confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions on test set
# Print confusion matrix and see which gestures are confused
```

---

### 8. Data File Issues

#### Problem: `CSV files corrupted or incomplete`

**Symptoms:**
- Model training fails
- Very few rows in CSV (~100 instead of ~9000)

**Debug:**
```python
import pandas as pd

for gesture in ["rest", "open", "fist", "strong_fist", "wrist_up", "wrist_down"]:
    df = pd.read_csv(f"data_raw/{gesture}.csv")
    print(f"{gesture}: {len(df)} rows (expected ~9000)")
    
    # Check for NaN values
    print(f"  NaN values: {df.isnull().sum().sum()}")
    
    # Check columns
    print(f"  Columns: {list(df.columns)}")
```

**Fix:** Re-run collection
```bash
python azmuth.py
```

---

### 9. Installation Issues

#### Problem: `pip install fails: Could not find a version`

**Common packages with issues:**

```bash
# torch - use official channel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# scikit-learn requires numpy first
pip install numpy scipy
pip install scikit-learn

# pyserial
pip install pyserial
```

#### Problem: `ModuleNotFoundError: No module named 'sklearn'`

**Fix:**
```bash
pip install scikit-learn
# NOT: pip install sklearn
```

---

## Quick Diagnostic Checklist

```
[ ] ESP32 connected via USB
[ ] Serial port correct (check COM3/COM5)
[ ] Baud rate 115200
[ ] Arduino code running on ESP32
[ ] Electrode contact good (clean, moist)
[ ] Sampling rate ~200 Hz (check azmuth.py output)
[ ] CSV files in data_raw/ (~9000 rows each)
[ ] graphs/ shows clear gesture patterns
[ ] Python 3.7+ installed
[ ] All packages installed (pip install -r requirements.txt)
[ ] Base model training shows accuracy >85%
[ ] Calibration completes without error
[ ] Real-time predictions show gesture names
```

---

## Getting Help

**If you're stuck:**

1. **Check the relevant script docstring**
   ```python
   python -c "import train_base_model; help(train_base_model)"
   ```

2. **Add debug prints to scripts**
   ```python
   # In calibrate_model.py, around line 100
   print(f"DEBUG: Calibration data shape: {X_calib.shape}")
   print(f"DEBUG: Labels: {y_calib}")
   ```

3. **Check console output carefully**
   - Look for ✅ (success) vs ⚠️ (warning) vs ❌ (error)
   - Error messages point to the problem

4. **Verify each step works independently**
   - Can you collect data? `python azmuth.py`
   - Can you train? `python train_base_model.py`
   - Can you calibrate? `python calibrate_model.py`
   - Can you predict? `python predict_realtime.py`

5. **Read the detailed documentation**
   - `README_FULL_SYSTEM.md` - Full system explanation
   - `ARCHITECTURE_DETAILED.md` - Data flow diagrams
   - Script headers - Code-level documentation

---

**Good luck! You've got this! 🎯**
