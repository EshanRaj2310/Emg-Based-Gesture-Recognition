"""
Fast calibration, model adaptation, benchmarking, and SIGNAL VISUALIZATION.
UPDATES:
1. SYNCED: Automatically uses the new 5-gesture list (No "Fist").
2. VISUALIZER: Real-time Signal Graph included.
3. ADAPTATION: Fine-tunes the base model for the specific user.
"""

import os
import re
import time
import serial
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import base model and configurations
# This will pull the updated GESTURES list (without "fist")
from train_base_model import EMGGestureNet, GESTURES, SAMPLING_RATE, DEVICE, MODEL_DIR, SAMPLES_PER_REP

# ============================================
# CONFIGURATION
# ============================================
ADAPTED_MODELS_DIR = os.path.join(MODEL_DIR, "adapted_models")
SENSOR_PORT = "COM5"  # ⚠️ CHECK YOUR PORT (e.g., /dev/ttyUSB0 or COM3)
SENSOR_BAUD = 115200

# Data Collection Settings
REPS_PER_GESTURE = 10
DURATION_PER_REP = 3
COLLECTION_DELAY = 1

# Fine-Tuning Hyperparameters
FT_EPOCHS = 50          
FT_BATCH_SIZE = 8
FT_LEARNING_RATE = 1e-5 

print(f"🔧 Using device: {DEVICE}")
print(f"📋 Calibrating for {len(GESTURES)} gestures: {GESTURES}")

# ============================================
# FUNCTION: REAL-TIME VISUALIZER
# ============================================

def visualize_live_signals(port=SENSOR_PORT, baud=SENSOR_BAUD):
    """
    Opens a live matplotlib window to inspect signals.
    Useful for checking sensor quality before recording.
    """
    print(f"\n📈 Opening Real-Time Signal Check on {port}...")
    print("Press 'Ctrl+C' in the terminal to STOP and return to menu.")
    
    digit_pattern = re.compile(r"(\d+)\D+(\d+)")
    
    # Buffers for plotting (last 500 points = ~2.5 seconds)
    window_size = 500
    ch1_data = deque([2000]*window_size, maxlen=window_size)
    ch2_data = deque([2000]*window_size, maxlen=window_size)
    
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Could not open port: {e}")
        return

    # Setup Plot
    plt.ion() # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Lines
    line1, = ax1.plot(ch1_data, color='blue', lw=1)
    line2, = ax2.plot(ch2_data, color='orange', lw=1)
    
    # Styling
    ax1.set_title("Channel 1 (Sensor 1)")
    ax1.set_ylim(0, 4096) # ESP32 12-bit range
    ax1.set_ylabel("Raw Value")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Channel 2 (Sensor 2)")
    ax2.set_ylim(0, 4096)
    ax2.set_ylabel("Raw Value")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = ""
    counter = 0
    
    try:
        while True:
            # 1. Read Serial Data
            if ser.in_waiting:
                try:
                    chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk
                    if '\n' in buffer:
                        lines = buffer.split('\n')
                        for line in lines[:-1]:
                            line = line.strip()
                            if not line: continue
                            match = digit_pattern.search(line)
                            if match:
                                ch1_data.append(int(match.group(1)))
                                ch2_data.append(int(match.group(2)))
                        buffer = lines[-1]
                except: pass
            
            # 2. Update Plot (every 5 iterations to save CPU)
            counter += 1
            if counter % 5 == 0:
                line1.set_ydata(ch1_data)
                line2.set_ydata(ch2_data)
                
                fig.canvas.flush_events()
                
            time.sleep(0.005) # Tiny sleep to prevent 100% CPU usage
            
    except KeyboardInterrupt:
        print("\n⏹️ Signal Check Stopped.")
    finally:
        plt.close(fig)
        ser.close()
        print("✅ Serial port released.")


# ============================================
# DATA COLLECTION (WITH DYNAMIC NORM)
# ============================================

def load_model_config() -> dict:
    config_path = os.path.join(MODEL_DIR, "model_config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f: return pickle.load(f)
    print("⚠️ Config not found, using defaults.")
    return {"use_features": False, "samples_per_rep": SAMPLES_PER_REP}

def collect_and_normalize(port: str = SENSOR_PORT, baud: int = SENSOR_BAUD, user_id: str = "01"):
    print("\n📊 Collecting user-specific data for calibration...")
    
    # Load config to ensure we match the training sample count
    config = load_model_config()
    samples_target = config.get("samples_per_rep", SAMPLES_PER_REP)
    digit_pattern = re.compile(r"(\d+)\D+(\d+)") 

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None, None

    raw_data = [] 
    labels = []
    
    # Iterate through the UPDATED gesture list (5 gestures)
    for gesture_idx, gesture in enumerate(GESTURES):
        print(f"\n🎯 Prepare for: {gesture.upper()}")
        input("Press Enter to start...")
        
        for rep in range(REPS_PER_GESTURE):
            print(f"  Repetition {rep+1}/{REPS_PER_GESTURE} - GO!")
            
            ch1_buf, ch2_buf = [], []
            ser.reset_input_buffer()
            buffer = "" 
            start_time = time.time()
            
            while (time.time() - start_time) < DURATION_PER_REP:
                if ser.in_waiting:
                    try:
                        chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        buffer += chunk
                        if '\n' in buffer:
                            lines = buffer.split('\n')
                            for line in lines[:-1]:
                                match = digit_pattern.search(line.strip())
                                if match:
                                    ch1_buf.append(int(match.group(1)))
                                    ch2_buf.append(int(match.group(2)))
                            buffer = lines[-1]
                    except: pass

            ch1_arr = np.array(ch1_buf)
            ch2_arr = np.array(ch2_buf)
            cnt = len(ch1_arr)
            
            # Ensure exact sample count matching the trained model
            if cnt < samples_target:
                pad = samples_target - cnt
                if pad > 0:
                    ch1_arr = np.pad(ch1_arr, (0, pad), mode='edge')
                    ch2_arr = np.pad(ch2_arr, (0, pad), mode='edge')
            else:
                ch1_arr = ch1_arr[:samples_target]
                ch2_arr = ch2_arr[:samples_target]
            
            raw_data.append(np.vstack([ch1_arr, ch2_arr]))
            labels.append(gesture_idx)
            
            print(f"    ✅ Collected {cnt} samples")
            if rep < REPS_PER_GESTURE - 1: time.sleep(COLLECTION_DELAY)
    
    ser.close()
    
    # Calculate & Save User Stats
    X_raw = np.array(raw_data) 
    print("\n🧮 Calculating USER-SPECIFIC normalization stats...")
    
    user_stats = {
        'mean_ch1': np.mean(X_raw[:, 0, :]),
        'std_ch1': np.std(X_raw[:, 0, :]),
        'mean_ch2': np.mean(X_raw[:, 1, :]),
        'std_ch2': np.std(X_raw[:, 1, :])
    }
    
    print(f"   CH1 Mean: {user_stats['mean_ch1']:.1f}")
    print(f"   CH2 Mean: {user_stats['mean_ch2']:.1f}")

    # Z-Score Normalization using user's own stats
    X_norm = np.zeros_like(X_raw, dtype=float)
    X_norm[:, 0, :] = (X_raw[:, 0, :] - user_stats['mean_ch1']) / (user_stats['std_ch1'] + 1e-8)
    X_norm[:, 1, :] = (X_raw[:, 1, :] - user_stats['mean_ch2']) / (user_stats['std_ch2'] + 1e-8)
    
    stats_path = os.path.join(ADAPTED_MODELS_DIR, f"user_{user_id}_normalization.pkl")
    os.makedirs(ADAPTED_MODELS_DIR, exist_ok=True)
    with open(stats_path, "wb") as f: pickle.dump(user_stats, f)
    print(f"💾 Normalization stats saved to: {stats_path}")

    return X_norm, np.array(labels)

# ============================================
# MODEL ADAPTATION
# ============================================

class UserEMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def adapt_and_benchmark(base_model_path: str, X_user: np.ndarray, y_user: np.ndarray, save_path: str):
    config = load_model_config()
    
    # Split calibration data
    X_train, X_test, y_train, y_test = train_test_split(
        X_user, y_user, test_size=0.2, random_state=42, stratify=y_user
    )
    
    # Initialize model with the NEW number of gestures (5)
    model = EMGGestureNet(num_gestures=len(GESTURES), use_features=config.get("use_features", False)).to(DEVICE)
    
    # Load weights
    # Since train_base_model.py was run with 5 gestures, the weights will match this architecture
    try:
        model.load_state_dict(torch.load(base_model_path, map_location=DEVICE))
        print("✅ Base model weights loaded successfully.")
    except RuntimeError as e:
        print(f"❌ Model mismatch error: {e}")
        print("💡 Hint: Ensure you re-ran 'train_base_model.py' after changing the gesture list.")
        return
    
    # Freeze Feature Extractor (Conv + LSTM), Train Classifier
    for name, param in model.named_parameters():
        if "fc" not in name: param.requires_grad = False
            
    train_loader = DataLoader(UserEMGDataset(X_train, y_train), batch_size=FT_BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🔧 Fine-tuning model...")
    model.train()
    for epoch in range(FT_EPOCHS):
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{FT_EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    
    # Benchmark
    print("\n🧪 BENCHMARK RESULTS")
    test_loader = DataLoader(UserEMGDataset(X_test, y_test), batch_size=len(X_test))
    model.eval()
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(DEVICE)
            outputs = model(X_b)
            _, preds = torch.max(outputs, 1)
            acc = accuracy_score(y_b.numpy(), preds.cpu().numpy()) * 100
            print(classification_report(y_b.numpy(), preds.cpu().numpy(), target_names=GESTURES, zero_division=0))
            print(f"🏆 Personal Accuracy: {acc:.2f}%")

# ============================================
# MAIN
# ============================================

def main():
    print("="*50 + "\n   CALIBRATION & VISUALIZATION TOOL\n" + "="*50)
    
    user_id = input("Enter user ID (e.g. '01'): ").strip() or "01"

    # Step 1: Optional Signal Check
    check = input("📈 Do you want to visualize signals first? (y/n): ").strip().lower()
    if check == 'y':
        visualize_live_signals()
        print("\nRe-starting calibration process...")
    
    # Step 2: Calibration
    base_path = os.path.join(MODEL_DIR, "base_model_best.pth")
    if not os.path.exists(base_path): base_path = os.path.join(MODEL_DIR, "base_model_final.pth")
    
    if not os.path.exists(base_path):
        print(f"❌ Base model not found at {base_path}. Please run 'train_base_model.py' first.")
        return

    # Collect data for the 5 gestures
    X, y = collect_and_normalize(user_id=user_id)
    if X is None: return

    out_path = os.path.join(ADAPTED_MODELS_DIR, f"user_{user_id}_model.pth")
    adapt_and_benchmark(base_path, X, y, out_path)
    
    print(f"\n✅ DONE! Use 'user_{user_id}_model.pth' and 'user_{user_id}_normalization.pkl' for realtime.")

if __name__ == "__main__":
    main()