"""
Real-Time EMG Inference Script
==============================
1. Connects to ESP32 via Serial.
2. Loads the trained model (Base or User-Adapted).
3. Predicts gestures in real-time.
4. Uses 'Majority Voting' to smooth outputs.
"""

import os
import re
import time
import serial
import torch
import numpy as np
import pickle
import collections
import argparse

# Import model architecture
from train_base_model import EMGGestureNet, DEVICE

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = "models"
ADAPTED_MODELS_DIR = os.path.join(MODEL_DIR, "adapted_models")
SERIAL_PORT = "COM5"  # ⚠️ UPDATE THIS
BAUD_RATE = 115200

# Inference Settings
WINDOW_SIZE = 600       # Must match training (3s * 200Hz) ?? 
# actually, the training used 3s windows, but for realtime 
# we usually want a sliding window. 
# However, the model architecture (CNN+LSTM) expects a specific input length (600).
# We will use a rolling buffer of 600 samples.

PREDICTION_INTERVAL = 0.1  # Predict every 100ms
SMOOTHING_BUFFER = 5       # Number of predictions to keep for majority vote
CONFIDENCE_THRESHOLD = 0.7 # Only accept predictions with high probability

# ============================================
# HELPER CLASSES
# ============================================

class DataBuffer:
    """Thread-safe-ish circular buffer for serial data."""
    def __init__(self, channels=2, max_len=600):
        self.max_len = max_len
        self.buffer = collections.deque(maxlen=max_len)
        self.channels = channels

    def add(self, sample):
        """Add a [ch1, ch2] sample."""
        if len(sample) == self.channels:
            self.buffer.append(sample)

    def is_ready(self):
        return len(self.buffer) == self.max_len

    def get_window(self):
        """Return numpy array of shape (2, max_len)."""
        # Convert to numpy: (max_len, 2)
        arr = np.array(self.buffer)
        # Transpose to match model input: (2, max_len)
        return arr.T

class Smoother:
    """Smoothing filter to reduce prediction flickering."""
    def __init__(self, history_size=5):
        self.history = collections.deque(maxlen=history_size)

    def update(self, prediction_idx):
        self.history.append(prediction_idx)
        # Find most common element
        if len(self.history) > 0:
            count = collections.Counter(self.history)
            most_common, num = count.most_common(1)[0]
            return most_common
        return prediction_idx

# ============================================
# MAIN FUNCTIONS
# ============================================

def load_config():
    try:
        with open(os.path.join(MODEL_DIR, "model_config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Load gesture mappings
        with open(os.path.join(MODEL_DIR, "gesture_mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
            
        return config, mappings
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None, None

def load_normalization(user_id=None):
    """Load normalization parameters (User specific or Global)."""
    if user_id:
        path = os.path.join(ADAPTED_MODELS_DIR, f"user_{user_id}_normalization.pkl")
        if os.path.exists(path):
            print(f"✅ Loading user {user_id} normalization stats.")
            with open(path, "rb") as f: return pickle.load(f)
            
    # Fallback to base model norm
    path = os.path.join(MODEL_DIR, "normalization_params.pkl")
    if os.path.exists(path):
        print("✅ Loading global normalization stats.")
        with open(path, "rb") as f: return pickle.load(f)
        
    print("⚠️ No normalization stats found! Predictions may be wrong.")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default=None, help="User ID for adapted model (e.g., '01')")
    args = parser.parse_args()

    # 1. Load Config & Mappings
    config, mappings = load_config()
    if not config: return
    
    gestures = config['gestures']
    idx_to_gesture = mappings['idx_to_gesture']
    
    print(f"📋 Gestures: {gestures}")

    # 2. Load Model
    model = EMGGestureNet(num_gestures=len(gestures), use_features=config['use_features'])
    
    # Check for user model first, then base model
    model_path = ""
    if args.user:
        user_path = os.path.join(ADAPTED_MODELS_DIR, f"user_{args.user}_model.pth")
        if os.path.exists(user_path):
            model_path = user_path
            print(f"🧠 Loading ADAPTED model: {os.path.basename(model_path)}")
        else:
            print(f"⚠️ User model not found. Falling back to base.")
    
    if not model_path:
        model_path = os.path.join(MODEL_DIR, "base_model_final.pth")
        print(f"🧠 Loading BASE model: {os.path.basename(model_path)}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Load Normalization
    norm_params = load_normalization(args.user)
    
    # 4. Setup Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"🔌 Connected to {SERIAL_PORT}")
        time.sleep(2) # Allow Arduino reset
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Serial error: {e}")
        return

    # 5. Runtime Loop
    buffer = DataBuffer(max_len=config['samples_per_rep']) # 600 samples
    smoother = Smoother(history_size=SMOOTHING_BUFFER)
    digit_pattern = re.compile(r"(\d+)\D+(\d+)")
    
    print("\n🚀 STARTING INFERENCE... (Ctrl+C to stop)\n")
    print(f"{'PREDICTION':<15} | {'CONFIDENCE':<10} | {'STATUS'}")
    print("-" * 40)

    last_pred_time = time.time()
    
    try:
        while True:
            # --- Read Serial ---
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    match = digit_pattern.search(line)
                    if match:
                        ch1 = int(match.group(1))
                        ch2 = int(match.group(2))
                        buffer.add([ch1, ch2])
                except:
                    pass

            # --- Predict Periodically ---
            current_time = time.time()
            if buffer.is_ready() and (current_time - last_pred_time > PREDICTION_INTERVAL):
                
                # Get window (2, 600)
                raw_window = buffer.get_window()
                
                # Normalize
                if norm_params:
                    # Check if using global or user params structure
                    if 'mean_ch1' in norm_params: # User structure
                        mean1, std1 = norm_params['mean_ch1'], norm_params['std_ch1']
                        mean2, std2 = norm_params['mean_ch2'], norm_params['std_ch2']
                    else: # Base structure (Global)
                        mean1, std1 = norm_params['global_mean_ch1'], norm_params['global_std_ch1']
                        mean2, std2 = norm_params['global_mean_ch2'], norm_params['global_std_ch2']
                    
                    # Apply Norm
                    norm_window = np.zeros_like(raw_window, dtype=float)
                    norm_window[0, :] = (raw_window[0, :] - mean1) / (std1 + 1e-8)
                    norm_window[1, :] = (raw_window[1, :] - mean2) / (std2 + 1e-8)
                else:
                    norm_window = raw_window

                # Convert to Tensor (Batch size 1)
                # Shape: (1, 2, 600)
                input_tensor = torch.from_numpy(norm_window).float().unsqueeze(0).to(DEVICE)

                # Inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                pred_idx = predicted.item()
                conf_val = confidence.item()

                # Smoothing
                final_idx = smoother.update(pred_idx)
                final_gesture = idx_to_gesture[final_idx]

                # Visual Output
                # Only print if confidence is decent, otherwise print "..."
                display_gesture = final_gesture.upper() if conf_val > 0.5 else "..."
                
                # Clear line and print
                print(f"\r{display_gesture:<15} | {conf_val*100:.1f}%      |", end="")
                
                last_pred_time = current_time

    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped.")
        ser.close()

if __name__ == "__main__":
    main()