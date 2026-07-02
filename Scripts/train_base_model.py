"""
Train base EMG gesture recognition model from multi-session data.

This script:
1. Loads 7 sessions of EMG data for 5 gestures (collected at 200Hz)
2. Extracts features from each 3-second repetition (RMS, mean, variance, etc.)
3. Creates a CNN + LSTM architecture to capture temporal patterns
4. Trains the model to classify gestures robustly across sessions
5. Saves the trained base model for later fine-tuning

The model is designed to learn gesture patterns that generalize across:
- Multiple repetitions within a session
- Multiple sessions from the same user
- Variations in electrode positioning and skin impedance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List
import pickle

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = "data_raw"
MODEL_DIR = "models"
SAMPLING_RATE = 200  # Hz
DURATION_PER_REP = 3  # seconds
SAMPLES_PER_REP = SAMPLING_RATE * DURATION_PER_REP  # 600 samples

# UPDATED: Removed "fist" from this list
GESTURES = ["rest", "open", "strong_fist", "wrist_up", "wrist_down"]

# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Feature extraction window for sliding window (useful for real-time prediction)
WINDOW_SIZE = 100  # ~0.5 seconds
WINDOW_STRIDE = 50  # 50% overlap

# Data augmentation settings
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 3  # Multiply dataset size by this factor

# Feature extraction settings
USE_FEATURES = False  # Set to True to use extracted features instead of raw data

print(f"🔧 Using device: {DEVICE}")
print(f"🎯 Training on {len(GESTURES)} gestures: {GESTURES}")


# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_time_domain_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract time-domain features from EMG signal.
    
    Features:
    - RMS (Root Mean Square): energy measure
    - Mean Absolute Value (MAV): amplitude measure
    - Variance: signal variability
    - Waveform Length: sum of absolute differences
    - Zero Crossings: noise/activity indicator
    """
    rms = np.sqrt(np.mean(signal ** 2))
    mav = np.mean(np.abs(signal))
    variance = np.var(signal)
    waveform_length = np.sum(np.abs(np.diff(signal)))
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2
    
    return np.array([rms, mav, variance, waveform_length, zero_crossings])


def extract_freq_domain_features(signal: np.ndarray, sr: int = SAMPLING_RATE) -> np.ndarray:
    """Extract frequency-domain features using FFT."""
    fft = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sr)
    
    # Only positive frequencies
    positive_fft = fft[:len(fft)//2]
    positive_freqs = freqs[:len(freqs)//2]
    
    # Mean frequency, power spectrum features
    mean_freq = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft) if np.sum(positive_fft) > 0 else 0
    median_freq = positive_freqs[np.argmax(np.cumsum(positive_fft) >= np.sum(positive_fft) / 2)]
    power = np.sum(positive_fft ** 2)
    
    return np.array([mean_freq, median_freq, power])


def extract_features_from_rep(ch1: np.ndarray, ch2: np.ndarray) -> np.ndarray:
    """
    Extract features from a single 3-second repetition (2 channels).
    
    Returns: feature vector combining time and frequency domain features
    """
    # Time-domain features for each channel
    td_ch1 = extract_time_domain_features(ch1)
    td_ch2 = extract_time_domain_features(ch2)
    
    # Frequency-domain features for each channel
    fd_ch1 = extract_freq_domain_features(ch1)
    fd_ch2 = extract_freq_domain_features(ch2)
    
    # Combined feature vector
    features = np.concatenate([td_ch1, td_ch2, fd_ch1, fd_ch2])
    return features


def get_sliding_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Create sliding window views of signal for CNN input."""
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


# ============================================
# DATA AUGMENTATION
# ============================================

def augment_emg_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to EMG signals."""
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Original sample
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Apply augmentation AUGMENTATION_FACTOR-1 times
        for _ in range(AUGMENTATION_FACTOR - 1):
            # Randomly choose augmentation technique
            aug_type = np.random.choice(['noise', 'shift', 'scale'])
            
            if aug_type == 'noise':
                # Add noise
                noise = np.random.normal(0, 0.01, X[i].shape)
                augmented_X.append(X[i] + noise)
                augmented_y.append(y[i])
                
            elif aug_type == 'shift':
                # Time shift (small random shift)
                shift = np.random.randint(-10, 10)
                shifted = np.zeros_like(X[i])
                if shift > 0:
                    shifted[:, shift:] = X[i][:, :-shift]
                elif shift < 0:
                    shifted[:, :shift] = X[i][:, -shift:]
                else:
                    shifted = X[i]
                augmented_X.append(shifted)
                augmented_y.append(y[i])
                
            elif aug_type == 'scale':
                # Amplitude scaling
                scale = np.random.uniform(0.9, 1.1)
                augmented_X.append(X[i] * scale)
                augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)


# ============================================
# DATASET & DATALOADER
# ============================================

class EMGDataset(Dataset):
    """Custom Dataset for EMG gesture recognition."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# NEURAL NETWORK MODEL
# ============================================

class EMGGestureNet(nn.Module):
    """
    CNN + LSTM architecture for EMG gesture recognition.
    
    - CNN layers: Extract spatial patterns from raw EMG signals
    - LSTM layers: Capture temporal patterns across the gesture duration
    - Fully connected layers: Classification
    """
    
    def __init__(self, num_gestures: int, use_features: bool = False):
        super(EMGGestureNet, self).__init__()
        self.use_features = use_features
        
        if use_features:
            # For feature-based input
            self.fc1 = nn.Linear(16, 64)  # 16 features (8 per channel)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_gestures)
        else:
            # CNN for raw signal processing (treat 2-channel EMG as 2D data)
            self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool = nn.MaxPool1d(2)
            
            # LSTM for temporal dynamics
            self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.3)
            
            # Fully connected layers
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, num_gestures)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if self.use_features:
            # Feature-based processing
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        else:
            # CNN feature extraction
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            
            # Prepare for LSTM (B, C, L) -> (B, L, C)
            x = x.transpose(1, 2)
            
            # LSTM processing
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use last hidden state for classification
            x = h_n[-1]  # (B, 64)
            
            # Classification head
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        
        return x


# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

def load_gesture_data(gesture: str, data_dir: str = DATA_DIR) -> Tuple[np.ndarray, List[str]]:
    """
    Load all CSV data for a single gesture across ALL session subfolders.
    Scans data_raw/session_01, data_raw/session_02, etc.
    
    Returns:
        X: numpy array of shape (num_reps, 2, SAMPLES_PER_REP) or (num_reps, 16) if using features
        session_ids: list of session identifiers for each repetition
    """
    all_reps_list = []
    session_ids = []
    
    # 1. Find all session folders
    if not os.path.exists(data_dir):
        print(f"⚠️ Error: {data_dir} does not exist")
        return np.empty((0, 2, SAMPLES_PER_REP)), []
        
    # Get list of all items in data_dir, filter for directories starting with 'session_'
    items = sorted(os.listdir(data_dir))
    session_folders = [d for d in items if os.path.isdir(os.path.join(data_dir, d))]
    
    # If no subfolders found, try loading from root (backward compatibility)
    if not session_folders:
        session_folders = ["."]
        
    print(f"   Searching {len(session_folders)} folders for '{gesture}'...")

    # 2. Iterate through every session
    for folder in session_folders:
        csv_path = os.path.join(data_dir, folder, f"{gesture}.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path)
            
            # Basic validation
            if "adc_ch1" not in df.columns or "rep" not in df.columns:
                continue

            # Extract ADC values
            ch1 = df["adc_ch1"].values
            ch2 = df["adc_ch2"].values
            reps = df["rep"].values
            unique_reps = np.unique(reps)
            
            for rep in unique_reps:
                mask = reps == rep
                rep_ch1 = ch1[mask]
                rep_ch2 = ch2[mask]
                
                # Pad or trim to exact SAMPLES_PER_REP (600)
                if len(rep_ch1) < SAMPLES_PER_REP:
                    pad_len = SAMPLES_PER_REP - len(rep_ch1)
                    rep_ch1 = np.pad(rep_ch1, (0, pad_len), mode='edge')
                    rep_ch2 = np.pad(rep_ch2, (0, pad_len), mode='edge')
                else:
                    rep_ch1 = rep_ch1[:SAMPLES_PER_REP]
                    rep_ch2 = rep_ch2[:SAMPLES_PER_REP]
                
                if USE_FEATURES:
                    # Extract features instead of using raw data
                    features = extract_features_from_rep(rep_ch1, rep_ch2)
                    all_reps_list.append(features)
                else:
                    # Stack channels (2, 600)
                    rep_data = np.vstack([rep_ch1, rep_ch2])
                    all_reps_list.append(rep_data)
                
                # Track session ID for this repetition
                session_ids.append(folder)
                
        except Exception as e:
            print(f"    ⚠️ Error reading {csv_path}: {e}")

    if len(all_reps_list) == 0:
        if USE_FEATURES:
            return np.empty((0, 16)), []
        else:
            return np.empty((0, 2, SAMPLES_PER_REP)), []
    
    # Stack all repetitions
    if USE_FEATURES:
        return np.array(all_reps_list), session_ids
    else:
        return np.array(all_reps_list), session_ids


def normalize_session_data(session_data: np.ndarray, session_ids: List[str]) -> Tuple[np.ndarray, dict]:
    """
    Normalize EMG data per session to maintain consistency.
    
    Args:
        session_data: numpy array of shape (num_reps, 2, SAMPLES_PER_REP)
        session_ids: list of session identifiers for each repetition
    
    Returns:
        normalized_data: normalized EMG data
        norm_params: dictionary with normalization parameters
    """
    # Group data by session
    sessions = {}
    for i, session_id in enumerate(session_ids):
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(i)
    
    # Initialize normalized data
    normalized_data = np.zeros_like(session_data)
    
    # Store normalization parameters for each session
    norm_params = {
        'session_params': {},
        'global_mean_ch1': 0,
        'global_std_ch1': 1,
        'global_mean_ch2': 0,
        'global_std_ch2': 1
    }
    
    # Calculate global statistics across all sessions
    all_ch1 = session_data[:, 0, :].flatten()
    all_ch2 = session_data[:, 1, :].flatten()
    norm_params['global_mean_ch1'] = np.mean(all_ch1)
    norm_params['global_std_ch1'] = np.std(all_ch1)
    norm_params['global_mean_ch2'] = np.mean(all_ch2)
    norm_params['global_std_ch2'] = np.std(all_ch2)
    
    # Normalize each session independently
    for session_id, indices in sessions.items():
        # Get all data from this session
        session_reps = session_data[indices]
        
        # Flatten all data from the session
        ch1_data = session_reps[:, 0, :].flatten()
        ch2_data = session_reps[:, 1, :].flatten()
        
        # Calculate session-specific statistics
        ch1_mean, ch1_std = np.mean(ch1_data), np.std(ch1_data)
        ch2_mean, ch2_std = np.mean(ch2_data), np.std(ch2_data)
        
        # Store normalization parameters
        norm_params['session_params'][session_id] = {
            'ch1_mean': ch1_mean,
            'ch1_std': ch1_std,
            'ch2_mean': ch2_mean,
            'ch2_std': ch2_std
        }
        
        # Normalize each repetition using session statistics
        for i, rep_idx in enumerate(indices):
            ch1 = session_data[rep_idx, 0, :]
            ch2 = session_data[rep_idx, 1, :]
            
            # Z-score normalization
            ch1_norm = (ch1 - ch1_mean) / (ch1_std + 1e-8)
            ch2_norm = (ch2 - ch2_mean) / (ch2_std + 1e-8)
            
            normalized_data[rep_idx, 0, :] = ch1_norm
            normalized_data[rep_idx, 1, :] = ch2_norm
    
    return normalized_data, norm_params


def load_all_data() -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load all gesture data and create train/test split."""
    all_X = []
    all_y = []
    all_session_ids = []
    
    print("\n📂 Loading EMG data from multi-session folders...")
    for gesture_idx, gesture in enumerate(GESTURES):
        X_gesture, session_ids = load_gesture_data(gesture)
        
        if len(X_gesture) > 0:
            print(f"✅ {gesture}: {len(X_gesture)} total repetitions loaded")
            all_X.append(X_gesture)
            all_y.extend([gesture_idx] * len(X_gesture))
            all_session_ids.extend(session_ids)
        else:
            print(f"⚠️ {gesture}: No data found in any session folder")
    
    if len(all_X) == 0:
        raise ValueError("No data loaded! Check 'data_raw/' structure.")
    
    X = np.vstack(all_X)  # (total_reps, 2, 600) or (total_reps, 16) if using features
    y = np.array(all_y)
    
    # Normalize data if using raw EMG signals
    if not USE_FEATURES:
        X, norm_params = normalize_session_data(X, all_session_ids)
        print(f"\n✅ Data normalized per session")
    else:
        # For feature-based approach, use standard scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        norm_params = {
            'scaler': scaler,
            'global_mean_ch1': 0,
            'global_std_ch1': 1,
            'global_mean_ch2': 0,
            'global_std_ch2': 1
        }
    
    print(f"\n✅ Dataset Ready: {len(X)} samples, {len(np.unique(y))} classes")
    return X, y, norm_params


# ============================================
# TRAINING
# ============================================

def train_model(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE):
    """Train the neural network."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
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
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "base_model_best.pth"))
    
    return train_losses, val_losses, val_accs


def evaluate_model(model, test_loader, device=DEVICE):
    """Evaluate model with detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=GESTURES)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GESTURES, yticklabels=GESTURES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    
    # Print classification report
    print("\nClassification Report:")
    print(report)
    
    return cm, report


def plot_training_history(train_losses, val_losses, val_accs):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_accs, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=150)
    print("📊 Training history saved to: training_history.png")
    plt.show()


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("EMG GESTURE RECOGNITION - BASE MODEL TRAINING")
    print("=" * 60)
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load data
    X, y, norm_params = load_all_data()
    
    # Apply data augmentation if enabled
    if USE_AUGMENTATION and not USE_FEATURES:
        print(f"\n🔄 Applying data augmentation (factor: {AUGMENTATION_FACTOR})...")
        X, y = augment_emg_data(X, y)
        print(f"✅ Dataset after augmentation: {len(X)} samples")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Further split train into train/val (75/25 of training data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
    )
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Create datasets and dataloaders
    train_dataset = EMGDataset(X_train, y_train)
    val_dataset = EMGDataset(X_val, y_val)
    test_dataset = EMGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print(f"\n🧠 Creating model...")
    # NOTE: num_gestures is calculated automatically from len(GESTURES)
    model = EMGGestureNet(num_gestures=len(GESTURES), use_features=USE_FEATURES).to(DEVICE)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    train_losses, val_losses, val_accs = train_model(model, train_loader, val_loader, epochs=EPOCHS)
    
    # Test
    print(f"\n🧪 Testing on held-out test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    test_acc = 100 * correct / total
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    
    # Detailed evaluation
    evaluate_model(model, test_loader)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "base_model_final.pth"))
    print(f"💾 Model saved to: {os.path.join(MODEL_DIR, 'base_model_final.pth')}")
    
    # Save normalization parameters
    with open(os.path.join(MODEL_DIR, "normalization_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)
    print(f"💾 Normalization parameters saved to: {os.path.join(MODEL_DIR, 'normalization_params.pkl')}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accs)
    
    # Save gesture mappings for later use
    with open(os.path.join(MODEL_DIR, "gesture_mappings.pkl"), "wb") as f:
        pickle.dump({"gesture_to_idx": {g: i for i, g in enumerate(GESTURES)},
                     "idx_to_gesture": {i: g for i, g in enumerate(GESTURES)}}, f)
    
    # Save model configuration
    config = {
        "use_features": USE_FEATURES,
        "sampling_rate": SAMPLING_RATE,
        "duration_per_rep": DURATION_PER_REP,
        "samples_per_rep": SAMPLES_PER_REP,
        "gestures": GESTURES,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE
    }
    with open(os.path.join(MODEL_DIR, "model_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    print("\n" + "=" * 60)
    print("✅ Base model training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()