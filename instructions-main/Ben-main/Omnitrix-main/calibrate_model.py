"""
Adapt the base EMG model to a specific user with minimal data.

This script:
1. Loads the base model
2. Collects a small amount of user-specific data
3. Fine-tunes ONLY the final layers of the model with a low learning rate
4. Saves the adapted model for real-time use
"""

import os
import re
import time
import serial
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

# Import base model and configurations
from train_base_model import EMGGestureNet, GESTURES, SAMPLING_RATE, DEVICE, MODEL_DIR, SAMPLES_PER_REP

# ============================================
# CONFIGURATION
# ============================================
ADAPTED_MODELS_DIR = os.path.join(MODEL_DIR, "adapted_models")

# Serial connection
SENSOR_PORT = "COM5"  # Change to your port
SENSOR_BAUD = 115200

# Data collection settings
REPS_PER_GESTURE = 5  # INCREASED: More data helps fine-tuning
DURATION_PER_REP = 3  # seconds
COLLECTION_DELAY = 1  # seconds between gestures

# Fine-tuning settings
FT_EPOCHS = 5  # DECREASED: Fewer epochs to prevent overfitting on small data
FT_BATCH_SIZE = 8
FT_LEARNING_RATE = 0.0005  # Slightly higher LR for the small, unfrozen layers

# Real-time processing
WINDOW_SIZE = int(SAMPLING_RATE * 0.5)  # 0.5 second window
WINDOW_STRIDE = int(SAMPLING_RATE * 0.25)  # 0.25 second stride (4x overlap)

print(f"🔧 Using device: {DEVICE}")


# ============================================
# DATASET & DATALOADER
# ============================================

class UserEMGDataset(Dataset):
    """Dataset for user-specific EMG data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# HELPERS
# ============================================

def parse_dual_channel(line: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract two integers from serial line."""
    nums = re.findall(r"\d+", line)
    if len(nums) >= 2:
        try:
            return int(nums[0]), int(nums[1])
        except ValueError:
            return None, None
    return None, None


def load_model_config() -> dict:
    """Load model configuration."""
    config_path = os.path.join(MODEL_DIR, "model_config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            return pickle.load(f)
    else:
        # Default configuration
        return {
            "use_features": False,
            "sampling_rate": SAMPLING_RATE,
            "duration_per_rep": DURATION_PER_REP,
            "samples_per_rep": SAMPLES_PER_REP,
            "gestures": GESTURES,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE
        }


def load_normalization_params() -> dict:
    """Load normalization parameters saved during training."""
    norm_path = os.path.join(MODEL_DIR, "normalization_params.pkl")
    if os.path.exists(norm_path):
        with open(norm_path, "rb") as f:
            return pickle.load(f)
    else:
        # Default normalization parameters
        return {
            'session_params': {},
            'global_mean_ch1': 0,
            'global_std_ch1': 1,
            'global_mean_ch2': 0,
            'global_std_ch2': 1
        }


def normalize_channels(ch1: np.ndarray, ch2: np.ndarray, norm_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize EMG channels using saved parameters."""
    # Use global normalization parameters for user-specific data
    ch1_norm = (ch1 - norm_params['global_mean_ch1']) / (norm_params['global_std_ch1'] + 1e-8)
    ch2_norm = (ch2 - norm_params['global_mean_ch2']) / (norm_params['global_std_ch2'] + 1e-8)
    return ch1_norm, ch2_norm


def collect_user_data(port: str = SENSOR_PORT, baud: int = SENSOR_BAUD) -> Tuple[np.ndarray, np.ndarray]:
    """Collect user-specific EMG data for calibration."""
    print("\n📊 Collecting user-specific data for calibration...")
    print(f"Please perform each gesture for {DURATION_PER_REP} seconds when prompted.")
    print("Hold each gesture steadily, then relax.")
    
    # Load model configuration
    config = load_model_config()
    use_features = config.get("use_features", False)
    samples_per_rep = config.get("samples_per_rep", SAMPLES_PER_REP)
    
    # Connect to sensor
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(1)
        print(f"✅ Connected to {port} at {baud} baud")
    except Exception as e:
        print(f"❌ Could not connect to {port}: {e}")
        print("Using dummy data for demonstration...")
        return generate_dummy_data()
    
    # Collect data for each gesture
    all_data = []
    all_labels = []
    
    for gesture_idx, gesture in enumerate(GESTURES):
        print(f"\n🎯 Prepare for: {gesture.upper()}")
        print("Press Enter when ready to start collecting...")
        input()
        
        # Collect multiple repetitions of each gesture
        for rep in range(REPS_PER_GESTURE):
            print(f"  Repetition {rep+1}/{REPS_PER_GESTURE} - Start {gesture.upper()} NOW!")
            
            # Buffer to store data for this repetition
            ch1_buffer = []
            ch2_buffer = []
            
            # Collect data for specified duration
            start_time = time.time()
            while time.time() - start_time < DURATION_PER_REP:
                if ser.in_waiting:
                    raw_line = ser.readline().decode(errors="ignore").strip()
                    if not raw_line:
                        continue
                    
                    ch1, ch2 = parse_dual_channel(raw_line)
                    if ch1 is not None and ch2 is not None:
                        ch1_buffer.append(ch1)
                        ch2_buffer.append(ch2)
                
                time.sleep(0.001)
            
            # Convert to numpy arrays
            ch1_data = np.array(ch1_buffer)
            ch2_data = np.array(ch2_buffer)
            
            # Ensure we have enough data
            if len(ch1_data) < samples_per_rep:
                print(f"    ⚠️ Only collected {len(ch1_data)} samples, padding to {samples_per_rep}")
                pad_len = samples_per_rep - len(ch1_data)
                ch1_data = np.pad(ch1_data, (0, pad_len), mode='edge')
                ch2_data = np.pad(ch2_data, (0, pad_len), mode='edge')
            else:
                ch1_data = ch1_data[:samples_per_rep]
                ch2_data = ch2_data[:samples_per_rep]
            
            # Normalize using global parameters
            norm_params = load_normalization_params()
            ch1_norm, ch2_norm = normalize_channels(ch1_data, ch2_norm, norm_params)
            
            if use_features:
                # Extract features
                from train_base_model import extract_features_from_rep
                features = extract_features_from_rep(ch1_norm, ch2_norm)
                all_data.append(features)
            else:
                # Stack channels
                rep_data = np.vstack([ch1_norm, ch2_norm])
                all_data.append(rep_data)
            
            all_labels.append(gesture_idx)
            
            print(f"    ✅ Collected {len(ch1_data)} samples ({len(ch1_data)/DURATION_PER_REP:.1f} Hz)")
            
            # Short break between repetitions
            if rep < REPS_PER_GESTURE - 1:
                print(f"    Rest for {COLLECTION_DELAY} second(s)...")
                time.sleep(COLLECTION_DELAY)
    
    # Close connection
    ser.close()
    print("\n✅ Data collection complete!")
    
    return np.array(all_data), np.array(all_labels)


def generate_dummy_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate dummy data for demonstration when no sensor is available."""
    print("Generating dummy data for demonstration...")
    
    config = load_model_config()
    use_features = config.get("use_features", False)
    samples_per_rep = config.get("samples_per_rep", SAMPLES_PER_REP)
    
    all_data = []
    all_labels = []
    
    for gesture_idx, gesture in enumerate(GESTURES):
        for rep in range(REPS_PER_GESTURE):
            if use_features:
                # Generate random features
                features = np.random.randn(16)  # 16 features
                all_data.append(features)
            else:
                # Generate random EMG signals
                ch1 = np.random.randn(samples_per_rep) * 100
                ch2 = np.random.randn(samples_per_rep) * 100
                
                # Add some gesture-specific patterns
                if gesture != "rest":
                    # Add more energy for active gestures
                    ch1 += np.random.randn(samples_per_rep) * 200
                    ch2 += np.random.randn(samples_per_rep) * 200
                
                # Stack channels
                rep_data = np.vstack([ch1, ch2])
                all_data.append(rep_data)
            
            all_labels.append(gesture_idx)
    
    return np.array(all_data), np.array(all_labels)


def adapt_model(base_model_path: str, X_user: np.ndarray, y_user: np.ndarray, save_path: str):
    """
    Adapt the base model to user-specific data by freezing the feature extractor
    and only training the final classifier layers.
    """
    # Load model configuration
    config = load_model_config()
    use_features = config.get("use_features", False)
    
    # Load base model
    model = EMGGestureNet(num_gestures=len(GESTURES), use_features=use_features).to(DEVICE)
    model.load_state_dict(torch.load(base_model_path, map_location=DEVICE))
    
    # === KEY CHANGE: FREEZE THE FEATURE EXTRACTOR LAYERS ===
    print("🔒 Freezing feature extractor layers...")
    if not use_features:
        # Freeze Conv and LSTM layers
        for name, param in model.named_parameters():
            if "conv" in name or "lstm" in name:
                param.requires_grad = False
    else:
        # For feature-based models, freeze the first FC layer
        for name, param in model.named_parameters():
            if name == "fc1.weight" or name == "fc1.bias":
                param.requires_grad = False

    # We only want to train the parameters that are not frozen
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LEARNING_RATE)
    
    # Split user data into train and validation sets
    if len(X_user) < 4:  # If very little data, use all for training
        X_train, y_train = X_user, y_user
        X_val, y_val = X_user, y_user
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_user, y_user, test_size=0.25, random_state=42, stratify=y_user
        )
    
    # Create datasets and dataloaders
    train_dataset = UserEMGDataset(X_train, y_train)
    val_dataset = UserEMGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=FT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=FT_BATCH_SIZE, shuffle=False)
    
    # Fine-tune
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_EPOCHS)
    
    print(f"\n🔧 Fine-tuning only the final layers on {len(X_user)} samples...")
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(FT_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total if total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{FT_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
    
    # Save adapted model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Adapted model saved to: {save_path}")
    
    return model


# ============================================
# REAL-TIME TESTING
# ============================================

def test_adapted_model(model_path: str, port: str = SENSOR_PORT, duration: float = 30.0):
    """Test the adapted model in real-time."""
    print("\n" + "=" * 60)
    print("🎯 TESTING ADAPTED MODEL IN REAL-TIME")
    print("=" * 60)
    
    # Load model configuration
    config = load_model_config()
    use_features = config.get("use_features", False)
    
    # Load adapted model
    model = EMGGestureNet(num_gestures=len(GESTURES), use_features=use_features).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load normalization parameters
    norm_params = load_normalization_params()
    
    # Connect to sensor
    try:
        ser = serial.Serial(port, SENSOR_BAUD, timeout=0.1)
        time.sleep(1)
        print(f"✅ Connected to {port} at {SENSOR_BAUD} baud")
    except Exception as e:
        print(f"❌ Could not connect to {port}: {e}")
        return
    
    # Buffers for streaming data
    ch1_buffer = np.zeros(WINDOW_SIZE)
    ch2_buffer = np.zeros(WINDOW_SIZE)
    buffer_idx = 0
    buffer_full = False
    
    # Prediction history for smoothing
    pred_history = []
    
    try:
        print(f"Testing for {duration} seconds. Press Ctrl+C to stop early.")
        start_time = time.time()
        last_pred_time = 0
        
        while time.time() - start_time < duration:
            # Process incoming data
            if ser.in_waiting:
                raw_line = ser.readline().decode(errors="ignore").strip()
                if not raw_line:
                    continue
                
                ch1, ch2 = parse_dual_channel(raw_line)
                if ch1 is None or ch2 is None:
                    continue
                
                # Add to circular buffer
                ch1_buffer[buffer_idx] = ch1
                ch2_buffer[buffer_idx] = ch2
                
                buffer_idx = (buffer_idx + 1) % WINDOW_SIZE
                if buffer_idx == 0:
                    buffer_full = True
                
                # Make prediction at stride intervals
                current_time = time.time()
                if (buffer_full or buffer_idx >= WINDOW_SIZE) and (current_time - last_pred_time) >= (WINDOW_STRIDE / SAMPLING_RATE):
                    # Get window data
                    if buffer_full:
                        if buffer_idx + WINDOW_SIZE <= len(ch1_buffer):
                            # No wrap-around needed
                            window_ch1 = ch1_buffer[buffer_idx:buffer_idx + WINDOW_SIZE]
                            window_ch2 = ch2_buffer[buffer_idx:buffer_idx + WINDOW_SIZE]
                        else:
                            # Need to wrap around
                            end_part = len(ch1_buffer) - buffer_idx
                            window_ch1 = np.concatenate([
                                ch1_buffer[buffer_idx:],
                                ch1_buffer[:WINDOW_SIZE - end_part]
                            ])
                            window_ch2 = np.concatenate([
                                ch2_buffer[buffer_idx:],
                                ch2_buffer[:WINDOW_SIZE - end_part]
                            ])
                    else:
                        # Buffer not full yet
                        window_ch1 = ch1_buffer[:WINDOW_SIZE]
                        window_ch2 = ch2_buffer[:WINDOW_SIZE]
                    
                    # Normalize
                    window_ch1_norm, window_ch2_norm = normalize_channels(window_ch1, window_ch2, norm_params)
                    
                    # Prepare input
                    if use_features:
                        # Extract features
                        from train_base_model import extract_features_from_rep
                        features = extract_features_from_rep(window_ch1_norm, window_ch2_norm)
                        X = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
                    else:
                        # Stack channels
                        X = np.vstack([window_ch1_norm, window_ch2_norm])
                        X = torch.from_numpy(X).float().unsqueeze(0).to(DEVICE)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = model(X)
                        probs = torch.softmax(outputs, dim=1)
                        confidence, pred_idx = torch.max(probs, 1)
                    
                    pred_gesture = GESTURES[pred_idx.item()]
                    confidence = confidence.item()
                    
                    # Add to history for smoothing
                    pred_history.append((pred_gesture, confidence))
                    
                    # Keep only recent predictions
                    if len(pred_history) > 5:
                        pred_history.pop(0)
                    
                    # Get smoothed prediction
                    gesture_counts = {}
                    conf_sum = {}
                    for g, c in pred_history:
                        gesture_counts[g] = gesture_counts.get(g, 0) + 1
                        conf_sum[g] = conf_sum.get(g, 0) + c
                    
                    best_gesture = max(gesture_counts, key=gesture_counts.get)
                    avg_conf = conf_sum[best_gesture] / gesture_counts[best_gesture]
                    
                    # Display prediction
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] {best_gesture.upper():15s} ({avg_conf:.1%})")
                    
                    last_pred_time = current_time
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
    
    finally:
        ser.close()
        print("Disconnected from sensor")
        print("=" * 60)


# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "=" * 60)
    print("EMG GESTURE RECOGNITION - MODEL CALIBRATION")
    print("=" * 60)
    
    # Paths
    base_model_path = os.path.join(MODEL_DIR, "base_model_best.pth")
    os.makedirs(ADAPTED_MODELS_DIR, exist_ok=True)
    
    # Check if base model exists
    if not os.path.exists(base_model_path):
        print(f"\n❌ Base model not found: {base_model_path}")
        print("Please run the training script first to create the base model.")
        return
    
    # Get user identifier
    user_id = input("\nEnter user ID (e.g., '01'): ").strip()
    if not user_id:
        user_id = "default_user"
    
    # Collect user data
    X_user, y_user = collect_user_data()
    
    # Adapt model
    adapted_model_path = os.path.join(ADAPTED_MODELS_DIR, f"user_{user_id}_model.pth")
    model = adapt_model(base_model_path, X_user, y_user, adapted_model_path)
    
    # Ask if user wants to test the adapted model
    test_model = input("\nWould you like to test the adapted model in real-time? (y/n): ").strip().lower()
    if test_model == 'y':
        test_duration = input("Enter test duration in seconds (default: 30): ").strip()
        try:
            test_duration = float(test_duration) if test_duration else 30.0
        except ValueError:
            test_duration = 30.0
        
        test_adapted_model(adapted_model_path, duration=test_duration)
    
    print("\n" + "=" * 60)
    print("✅ Model calibration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()