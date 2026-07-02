"""
Real-time gesture prediction using adapted EMG model.

This script:
1. Loads the adapted model and its normalization parameters
2. Connects to EMG sensor via serial
3. Processes incoming EMG data in real-time
4. Predicts gestures and displays results
5. Handles streaming data with buffering and windowing
6. Provides visualization of predictions
7. Supports adaptive windowing based on confidence
"""

import os
import re
import time
import serial
import numpy as np
import pickle
import argparse
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn

# Import model architecture
from train_base_model import EMGGestureNet, GESTURES, SAMPLING_RATE, DEVICE

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = "models"
ADAPTED_MODELS_DIR = os.path.join(MODEL_DIR, "adapted_models")
DEFAULT_MODEL = "session_1_adapted_model.pth"

# Serial connection
SENSOR_PORT = "COM5"
SENSOR_BAUD = 115200

# Real-time processing
WINDOW_SIZE = int(SAMPLING_RATE * 0.5)  # 0.5 second window
WINDOW_STRIDE = int(SAMPLING_RATE * 0.25)  # 0.25 second stride (4x overlap)
CONFIDENCE_THRESHOLD = 0.6  # Min confidence for prediction
ADAPTIVE_THRESHOLD = 0.8  # High confidence threshold for adaptive windowing

# Visualization
SHOW_PLOT = True
PLOT_HISTORY = 50  # Number of predictions to show in plot

print(f"🔧 Using device: {DEVICE}")


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


def load_model_and_config(model_path: str) -> Tuple[nn.Module, Dict]:
    """Load the adapted model and its configuration."""
    # Load model configuration
    config_path = os.path.join(MODEL_DIR, "model_config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    else:
        # Default configuration
        config = {
            "use_features": False,
            "sampling_rate": SAMPLING_RATE,
            "gestures": GESTURES
        }
    
    # Load normalization parameters
    norm_path = os.path.join(MODEL_DIR, "normalization_params.pkl")
    if os.path.exists(norm_path):
        with open(norm_path, "rb") as f:
            norm_params = pickle.load(f)
    else:
        # Default normalization parameters
        norm_params = {
            'global_mean_ch1': 2000.0,
            'global_std_ch1': 500.0,
            'global_mean_ch2': 2000.0,
            'global_std_ch2': 500.0
        }
    
    # Load model
    model = EMGGestureNet(num_gestures=len(GESTURES), use_features=config.get("use_features", False)).to(DEVICE)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print(f"✅ Loaded model: {model_path}")
    return model, config, norm_params


def normalize_channels(ch1: np.ndarray, ch2: np.ndarray, norm_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize EMG channels using saved normalization parameters.
    """
    # Use global normalization parameters
    ch1_norm = (ch1 - norm_params['global_mean_ch1']) / (norm_params['global_std_ch1'] + 1e-8)
    ch2_norm = (ch2 - norm_params['global_mean_ch2']) / (norm_params['global_std_ch2'] + 1e-8)
    
    return ch1_norm, ch2_norm


def predict_gesture(model: nn.Module, ch1: np.ndarray, ch2: np.ndarray, 
                   norm_params: Dict, config: Dict) -> Tuple[str, float]:
    """
    Predict gesture from EMG window.
    
    Returns: (gesture_name, confidence)
    """
    # Normalize
    ch1_norm, ch2_norm = normalize_channels(ch1, ch2, norm_params)
    
    # Prepare input based on model configuration
    if config.get("use_features", False):
        # Extract features
        from train_base_model import extract_features_from_rep
        features = extract_features_from_rep(ch1_norm, ch2_norm)
        X = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    else:
        # Stack channels
        X = np.vstack([ch1_norm, ch2_norm])  # (2, window_size)
        X = torch.from_numpy(X).float().unsqueeze(0).to(DEVICE)  # (1, 2, window_size)
    
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    pred_gesture = GESTURES[pred_idx.item()]
    confidence = confidence.item()
    
    return pred_gesture, confidence


# ============================================
# VISUALIZATION
# ============================================

class GestureVisualizer:
    """Visualizer for real-time gesture predictions."""
    
    def __init__(self, history_size: int = PLOT_HISTORY):
        self.history_size = history_size
        self.prediction_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        
        # Set up the figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Real-time EMG Gesture Recognition', fontsize=16)
        
        # Initialize plots
        self.gesture_plot = None
        self.confidence_plot = None
        self.setup_plots()
        
        # Animation
        self.ani = None
        
    def setup_plots(self):
        """Set up the initial plots."""
        # Gesture prediction plot
        self.ax1.set_ylim(-0.5, len(GESTURES) - 0.5)
        self.ax1.set_yticks(range(len(GESTURES)))
        self.ax1.set_yticklabels(GESTURES)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_title('Gesture Predictions')
        self.ax1.grid(True, alpha=0.3)
        
        # Confidence plot
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Confidence')
        self.ax2.set_title('Prediction Confidence')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axhline(y=CONFIDENCE_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        
    def update(self, frame):
        """Update the plots with new data."""
        if not self.time_history:
            return self.gesture_plot, self.confidence_plot
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.setup_plots()
        
        # Convert gesture names to indices for plotting
        gesture_indices = [GESTURES.index(g) for g in self.prediction_history]
        
        # Plot gesture predictions
        self.gesture_plot = self.ax1.scatter(
            list(self.time_history), 
            gesture_indices, 
            c=list(self.confidence_history), 
            cmap='viridis', 
            vmin=0, 
            vmax=1,
            s=50
        )
        
        # Plot confidence
        self.confidence_plot = self.ax2.plot(
            list(self.time_history), 
            list(self.confidence_history), 
            'b-'
        )[0]
        
        # Add colorbar for the first frame
        if frame == 0:
            cbar = self.fig.colorbar(self.gesture_plot, ax=self.ax1)
            cbar.set_label('Confidence')
        
        return self.gesture_plot, self.confidence_plot
    
    def add_prediction(self, gesture: str, confidence: float, timestamp: float):
        """Add a new prediction to the history."""
        self.prediction_history.append(gesture)
        self.confidence_history.append(confidence)
        self.time_history.append(timestamp)
    
    def start_animation(self):
        """Start the animation."""
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update, 
            interval=100,  # Update every 100ms
            blit=False
        )
        plt.show()


# ============================================
# REAL-TIME PREDICTION ENGINE
# ============================================

class EMGPredictor:
    """Real-time EMG gesture predictor."""
    
    def __init__(self, model_path: str, port: str = SENSOR_PORT, baud: int = SENSOR_BAUD, 
                 show_plot: bool = SHOW_PLOT):
        self.model, self.config, self.norm_params = load_model_and_config(model_path)
        self.port = port
        self.baud = baud
        self.ser = None
        self.show_plot = show_plot
        
        # Buffers for streaming data
        self.buffer_size = WINDOW_SIZE + WINDOW_STRIDE  # Extra buffer for smooth sliding
        self.ch1_buffer = np.zeros(self.buffer_size)
        self.ch2_buffer = np.zeros(self.buffer_size)
        self.buffer_idx = 0
        self.buffer_full = False
        
        # Prediction history for smoothing
        self.pred_history = deque(maxlen=5)
        self.gesture_to_idx = {g: i for i, g in enumerate(GESTURES)}
        
        # Adaptive windowing
        self.current_stride = WINDOW_STRIDE
        self.last_confidence = 0.0
        
        # Visualization
        self.visualizer = GestureVisualizer() if show_plot else None
        self.plot_thread = None
        
        # Performance metrics
        self.prediction_times = deque(maxlen=100)
        self.start_time = time.time()
        
    def connect(self):
        """Connect to EMG sensor."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(1)
            print(f"✅ Connected to {self.port} at {self.baud} baud")
            return True
        except Exception as e:
            print(f"❌ Could not connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from sensor."""
        if self.ser:
            self.ser.close()
            print("Disconnected from sensor")
    
    def process_frame(self) -> bool:
        """
        Process one frame of data from serial.
        
        Returns: True if a complete window is ready for prediction
        """
        try:
            if self.ser.in_waiting:
                raw_line = self.ser.readline().decode(errors="ignore").strip()
                if not raw_line:
                    return False
                
                ch1, ch2 = parse_dual_channel(raw_line)
                if ch1 is None or ch2 is None:
                    return False
                
                # Add to circular buffer
                self.ch1_buffer[self.buffer_idx] = ch1
                self.ch2_buffer[self.buffer_idx] = ch2
                
                self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
                if self.buffer_idx == 0:
                    self.buffer_full = True
                
                # Check if we have enough data for a prediction
                return self.buffer_full or self.buffer_idx >= WINDOW_SIZE
            
            return False
        
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return False
    
    def get_window_data(self):
        """Get the most recent window of data."""
        if self.buffer_full:
            # Buffer is full, get the most recent WINDOW_SIZE samples
            if self.buffer_idx + WINDOW_SIZE <= self.buffer_size:
                # No wrap-around needed
                ch1 = self.ch1_buffer[self.buffer_idx:self.buffer_idx + WINDOW_SIZE]
                ch2 = self.ch2_buffer[self.buffer_idx:self.buffer_idx + WINDOW_SIZE]
            else:
                # Need to wrap around
                end_part = self.buffer_size - self.buffer_idx
                ch1 = np.concatenate([
                    self.ch1_buffer[self.buffer_idx:],
                    self.ch1_buffer[:WINDOW_SIZE - end_part]
                ])
                ch2 = np.concatenate([
                    self.ch2_buffer[self.buffer_idx:],
                    self.ch2_buffer[:WINDOW_SIZE - end_part]
                ])
        else:
            # Buffer not full yet
            ch1 = self.ch1_buffer[:WINDOW_SIZE]
            ch2 = self.ch2_buffer[:WINDOW_SIZE]
        
        return ch1, ch2
    
    def predict(self) -> Tuple[str, float]:
        """Get prediction from current buffer."""
        ch1, ch2 = self.get_window_data()
        
        # Measure prediction time
        pred_start = time.time()
        gesture, confidence = predict_gesture(self.model, ch1, ch2, self.norm_params, self.config)
        pred_time = time.time() - pred_start
        self.prediction_times.append(pred_time)
        
        # Update adaptive stride based on confidence
        if confidence > ADAPTIVE_THRESHOLD:
            # High confidence, use smaller stride for more responsive predictions
            self.current_stride = max(WINDOW_STRIDE // 2, 10)
        elif confidence < CONFIDENCE_THRESHOLD:
            # Low confidence, use larger stride to reduce noise
            self.current_stride = min(WINDOW_STRIDE * 2, WINDOW_SIZE)
        else:
            # Normal confidence, use default stride
            self.current_stride = WINDOW_STRIDE
        
        self.last_confidence = confidence
        
        # Add to history for smoothing
        self.pred_history.append((gesture, confidence))
        
        # Update visualization
        if self.visualizer:
            self.visualizer.add_prediction(gesture, confidence, time.time() - self.start_time)
        
        return gesture, confidence
    
    def get_smoothed_prediction(self) -> Tuple[str, float]:
        """Get smoothed prediction from history (weighted vote + avg confidence)."""
        if not self.pred_history:
            return "waiting", 0.0
        
        # Weighted vote on gesture (more recent predictions have higher weight)
        gesture_weights = {}
        for i, (gesture, conf) in enumerate(self.pred_history):
            weight = (i + 1) / len(self.pred_history)  # Linear weight based on recency
            gesture_weights[gesture] = gesture_weights.get(gesture, 0) + weight
        
        best_gesture = max(gesture_weights, key=gesture_weights.get)
        
        # Average confidence for best gesture
        confs = [conf for g, conf in self.pred_history if g == best_gesture]
        avg_conf = np.mean(confs)
        
        return best_gesture, avg_conf
    
    def start_visualization(self):
        """Start the visualization in a separate thread."""
        if self.visualizer:
            self.plot_thread = threading.Thread(target=self.visualizer.start_animation)
            self.plot_thread.daemon = True
            self.plot_thread.start()
    
    def run(self, duration: Optional[float] = None, verbose: bool = True):
        """
        Run continuous prediction.
        
        Args:
            duration: Run for N seconds (None = infinite)
            verbose: Print predictions
        """
        if not self.connect():
            return
        
        # Start visualization if enabled
        if self.show_plot:
            self.start_visualization()
            time.sleep(1)  # Give the plot time to initialize
        
        try:
            print("\n" + "=" * 60)
            print("🎯 REAL-TIME GESTURE PREDICTION")
            print("=" * 60)
            print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLING_RATE:.2f}s)")
            print(f"Default stride: {WINDOW_STRIDE} samples ({WINDOW_STRIDE/SAMPLING_RATE:.2f}s)")
            print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
            print(f"Adaptive threshold: {ADAPTIVE_THRESHOLD}")
            print("Press Ctrl+C to stop\n")
            
            start_time = time.time()
            frame_count = 0
            last_pred_time = 0
            last_stats_time = start_time
            
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    print("\n⏱️ Duration reached, stopping...")
                    break
                
                # Process incoming data
                ready = self.process_frame()
                
                # Make prediction at stride intervals
                if ready and (time.time() - last_pred_time) >= (self.current_stride / SAMPLING_RATE):
                    gesture, confidence = self.predict()
                    
                    # Get smoothed prediction
                    smoothed_gesture, smoothed_conf = self.get_smoothed_prediction()
                    
                    if smoothed_conf >= CONFIDENCE_THRESHOLD:
                        if verbose:
                            elapsed = time.time() - start_time
                            print(f"[{elapsed:.1f}s] {smoothed_gesture.upper():15s} ({smoothed_conf:.1%}) - Stride: {self.current_stride}")
                    
                    last_pred_time = time.time()
                    frame_count += 1
                
                # Print performance stats every 10 seconds
                if time.time() - last_stats_time > 10:
                    if self.prediction_times:
                        avg_pred_time = np.mean(self.prediction_times) * 1000  # ms
                        fps = frame_count / (time.time() - start_time)
                        print(f"📊 Performance: {fps:.1f} FPS, Avg prediction time: {avg_pred_time:.2f}ms")
                    last_stats_time = time.time()
                
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopped by user")
        
        finally:
            self.disconnect()
            print("=" * 60)


# ============================================
# COMMAND LINE INTERFACE
# ============================================

def list_available_models():
    """List all available adapted models."""
    print("\nAvailable models:")
    if os.path.exists(ADAPTED_MODELS_DIR):
        for f in os.listdir(ADAPTED_MODELS_DIR):
            if f.endswith(".pth"):
                print(f"  - {f}")
    else:
        print("  No adapted models found. Please run calibrate_model.py first.")


def main():
    parser = argparse.ArgumentParser(description='Real-time EMG gesture recognition')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Path to the adapted model (default: {DEFAULT_MODEL})')
    parser.add_argument('--port', type=str, default=SENSOR_PORT,
                        help=f'Serial port for EMG sensor (default: {SENSOR_PORT})')
    parser.add_argument('--baud', type=int, default=SENSOR_BAUD,
                        help=f'Baud rate for serial connection (default: {SENSOR_BAUD})')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration to run in seconds (default: infinite)')
    parser.add_argument('--list', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable real-time visualization')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("EMG GESTURE RECOGNITION - REAL-TIME PREDICTION")
    print("=" * 60)
    
    # List models and exit if requested
    if args.list:
        list_available_models()
        return
    
    # Find adapted model
    if not os.path.isabs(args.model):
        model_path = os.path.join(ADAPTED_MODELS_DIR, args.model)
    else:
        model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        list_available_models()
        print("\nPlease run: python calibrate_model.py")
        return
    
    # Create predictor
    predictor = EMGPredictor(
        model_path, 
        port=args.port, 
        baud=args.baud,
        show_plot=not args.no_plot
    )
    
    # Run continuous prediction
    predictor.run(duration=args.duration, verbose=True)


if __name__ == "__main__":
    main()