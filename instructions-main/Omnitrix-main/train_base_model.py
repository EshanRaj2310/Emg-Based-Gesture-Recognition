"""
Train base EMG gesture recognition model from multi-session data.

This script:
1. Loads 7 sessions of EMG data for 6 gestures (collected at 200Hz)
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
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = "data_raw"
MODEL_DIR = "models"
SAMPLING_RATE = 200  # Hz
DURATION_PER_REP = 3  # seconds
SAMPLES_PER_REP = SAMPLING_RATE * DURATION_PER_REP  # 600 samples

GESTURES = ["rest", "open", "fist", "strong_fist", "wrist_up", "wrist_down"]

# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Feature extraction window for sliding window (useful for real-time prediction)
WINDOW_SIZE = 100  # ~0.5 seconds
WINDOW_STRIDE = 50  # 50% overlap

print(f"🔧 Using device: {DEVICE}")


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
	
	def __init__(self, num_gestures: int = 6):
		super(EMGGestureNet, self).__init__()
		
		# CNN for raw signal processing (treat 2-channel EMG as 2D data)
		self.conv1 = nn.Conv1d(2, 32, kernel_size=5, padding=2)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
		self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
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

# ============================================
# DATA LOADING & PREPROCESSING (UPDATED)
# ============================================

def load_gesture_data(gesture: str, data_dir: str = DATA_DIR) -> np.ndarray:
    """
    Load all CSV data for a single gesture across ALL session subfolders.
    Scans data_raw/session_01, data_raw/session_02, etc.
    """
    all_reps_list = []
    
    # 1. Find all session folders
    if not os.path.exists(data_dir):
        print(f"⚠️ Error: {data_dir} does not exist")
        return np.empty((0, 2, SAMPLES_PER_REP))
        
    # Get list of all items in data_dir, filter for directories starting with 'session_'
    # Or just all directories if you prefer
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
                
                # Normalize (Z-score) per repetition to handle skin impedance changes
                rep_ch1 = (rep_ch1 - np.mean(rep_ch1)) / (np.std(rep_ch1) + 1e-8)
                rep_ch2 = (rep_ch2 - np.mean(rep_ch2)) / (np.std(rep_ch2) + 1e-8)
                
                # Stack channels (2, 600)
                rep_data = np.vstack([rep_ch1, rep_ch2])
                all_reps_list.append(rep_data)
                
        except Exception as e:
            print(f"    ⚠️ Error reading {csv_path}: {e}")

    if len(all_reps_list) == 0:
        return np.empty((0, 2, SAMPLES_PER_REP))
    
    # Stack all repetitions: (total_reps, 2, 600)
    return np.array(all_reps_list)


def load_all_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load all gesture data and create train/test split."""
    all_X = []
    all_y = []
    
    print("\n📂 Loading EMG data from multi-session folders...")
    for gesture_idx, gesture in enumerate(GESTURES):
        X_gesture = load_gesture_data(gesture)
        
        if len(X_gesture) > 0:
            print(f"✅ {gesture}: {len(X_gesture)} total repetitions loaded")
            all_X.append(X_gesture)
            all_y.extend([gesture_idx] * len(X_gesture))
        else:
            print(f"⚠️ {gesture}: No data found in any session folder")
    
    if len(all_X) == 0:
        raise ValueError("No data loaded! Check 'data_raw/' structure.")
    
    X = np.vstack(all_X)  # (total_reps, 2, 600)
    y = np.array(all_y)
    
    print(f"\n✅ Dataset Ready: {len(X)} samples, {len(np.unique(y))} classes")
    return X, y

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
	X, y = load_all_data()
	
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
	model = EMGGestureNet(num_gestures=len(GESTURES)).to(DEVICE)
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
	
	# Save final model
	torch.save(model.state_dict(), os.path.join(MODEL_DIR, "base_model_final.pth"))
	print(f"💾 Model saved to: {os.path.join(MODEL_DIR, 'base_model_final.pth')}")
	
	# Plot training history
	plot_training_history(train_losses, val_losses, val_accs)
	
	# Save gesture mappings for later use
	with open(os.path.join(MODEL_DIR, "gesture_mappings.pkl"), "wb") as f:
		pickle.dump({"gesture_to_idx": {g: i for i, g in enumerate(GESTURES)},
					 "idx_to_gesture": {i: g for i, g in enumerate(GESTURES)}}, f)
	
	print("\n" + "=" * 60)
	print("✅ Base model training complete!")
	print("=" * 60)


if __name__ == "__main__":
	main()
