"""
Real-Time EMG Inference (Stabilized)
====================================
1. Connects to ESP32.
2. Predicts gestures.
3. "Thinks" (Debounces) signal: Requires N consistent frames 
   before changing the displayed gesture.
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

from train_base_model import EMGGestureNet, DEVICE

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = "models"
ADAPTED_MODELS_DIR = os.path.join(MODEL_DIR, "adapted_models")
SERIAL_PORT = "COM5"  # ⚠️ UPDATE THIS IF NEEDED
BAUD_RATE = 115200

# STABILITY SETTINGS
PREDICTION_INTERVAL = 0.05   # Predict every 50ms (faster checks)
STABILITY_FRAMES = 6         # Must match this many times to "Lock In"
CONFIDENCE_THRESHOLD = 0.80  # Ignore anything below 80% certainty

# ============================================
# HELPER CLASSES
# ============================================

class DataBuffer:
    """Circular buffer for serial data."""
    def __init__(self, channels=2, max_len=600):
        self.max_len = max_len
        self.buffer = collections.deque(maxlen=max_len)

    def add(self, sample):
        self.buffer.append(sample)

    def is_ready(self):
        return len(self.buffer) == self.max_len

    def get_window(self):
        return np.array(self.buffer).T

class StabilityFilter:
    """
    The 'Think for a bit' Logic.
    Only changes state if the new gesture is held for 'required_frames'.
    """
    def __init__(self, required_frames=5):
        self.required_frames = required_frames
        self.current_display = "REST"  # What is currently locked in
        
        self.pending_gesture = None    # What we are currently thinking about
        self.counter = 0               # How long we've been thinking about it

    def update(self, new_gesture, confidence):
        # 1. Reject weak signals immediately
        if confidence < CONFIDENCE_THRESHOLD:
            # If signal is weak, do we reset? 
            # Ideally no, we just ignore this frame to prevent flickering gaps.
            return self.current_display, 0

        # 2. Check consistency
        if new_gesture == self.pending_gesture:
            self.counter += 1
        else:
            # Signal changed! Reset counter and start thinking about new gesture
            self.pending_gesture = new_gesture
            self.counter = 1

        # 3. Lock-in Logic
        if self.counter >= self.required_frames:
            self.current_display = new_gesture
            self.counter = self.required_frames # Cap the counter

        return self.current_display, self.counter

# ============================================
# SETUP & LOADERS
# ============================================

def load_config():
    try:
        with open(os.path.join(MODEL_DIR, "model_config.pkl"), "rb") as f:
            config = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "gesture_mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        return config, mappings
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None, None

def load_normalization(user_id=None):
    if user_id:
        path = os.path.join(ADAPTED_MODELS_DIR, f"user_{user_id}_normalization.pkl")
        if os.path.exists(path):
            print(f"✅ Loading user {user_id} normalization stats.")
            with open(path, "rb") as f: return pickle.load(f)
            
    path = os.path.join(MODEL_DIR, "normalization_params.pkl")
    if os.path.exists(path):
        print("✅ Loading global normalization stats.")
        with open(path, "rb") as f: return pickle.load(f)
    return None

# ============================================
# MAIN LOOP
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default=None)
    args = parser.parse_args()

    # 1. Setup
    config, mappings = load_config()
    if not config: return
    idx_to_gesture = mappings['idx_to_gesture']
    
    model = EMGGestureNet(num_gestures=len(config['gestures']), use_features=config['use_features'])
    
    # Load Weights
    model_path = os.path.join(MODEL_DIR, "base_model_final.pth")
    if args.user:
        user_path = os.path.join(ADAPTED_MODELS_DIR, f"user_{args.user}_model.pth")
        if os.path.exists(user_path): model_path = user_path
    
    print(f"🧠 Loading: {os.path.basename(model_path)}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    norm_params = load_normalization(args.user)
    
    # 2. Connect Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✅ Serial Connected.")
    except Exception as e:
        print(f"❌ Serial Error: {e}")
        return

    # 3. Initialize Logic
    buffer = DataBuffer(max_len=config['samples_per_rep'])
    stabilizer = StabilityFilter(required_frames=STABILITY_FRAMES)
    digit_pattern = re.compile(r"(\d+)\D+(\d+)")
    
    print(f"\n🛡️ STABILITY MODE: ON")
    print(f"   Required consecutive frames: {STABILITY_FRAMES}")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD*100}%\n")
    print(f"{'DETECTED':<15} | {'CONFIDENCE':<10} | {'STABILITY LOCK'}")
    print("-" * 50)

    last_pred_time = time.time()

    try:
        while True:
            # Fast Read
            if ser.in_waiting:
                lines = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').split('\n')
                for line in lines:
                    match = digit_pattern.search(line)
                    if match: buffer.add([int(match.group(1)), int(match.group(2))])

            # Periodic Predict
            if buffer.is_ready() and (time.time() - last_pred_time > PREDICTION_INTERVAL):
                
                # Prepare Data
                window = buffer.get_window()
                if norm_params:
                    # Handle both dict formats
                    if 'mean_ch1' in norm_params:
                        m1, s1 = norm_params['mean_ch1'], norm_params['std_ch1']
                        m2, s2 = norm_params['mean_ch2'], norm_params['std_ch2']
                    else:
                        m1, s1 = norm_params['global_mean_ch1'], norm_params['global_std_ch1']
                        m2, s2 = norm_params['global_mean_ch2'], norm_params['global_std_ch2']
                    
                    norm = np.zeros_like(window, dtype=float)
                    norm[0,:] = (window[0,:] - m1)/(s1+1e-8)
                    norm[1,:] = (window[1,:] - m2)/(s2+1e-8)
                else: norm = window

                # Predict
                inp = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(model(inp), dim=1)
                    conf, pred = torch.max(probs, 1)

                raw_gesture = idx_to_gesture[pred.item()]
                raw_conf = conf.item()

                # Stabilize ("Think")
                final_gesture, lock_level = stabilizer.update(raw_gesture, raw_conf)

                # Visualize
                # Create a progress bar [====  ] based on lock_level
                bars = "=" * lock_level
                spaces = " " * (STABILITY_FRAMES - lock_level)
                visual_lock = f"[{bars}{spaces}]"
                
                # Make text GREEN if locked, YELLOW if thinking
                status_color = "\033[92m" if lock_level == STABILITY_FRAMES else "\033[93m"
                reset_color = "\033[0m"

                print(f"\r{status_color}{final_gesture.upper():<15} | {raw_conf*100:.0f}%       | {visual_lock} {reset_color}", end="")
                
                last_pred_time = time.time()

    except KeyboardInterrupt:
        print("\n⏹️ Stopped.")
        ser.close()

if __name__ == "__main__":
    main()