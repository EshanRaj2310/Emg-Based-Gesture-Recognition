"""
ADVANCED PROJECT ANALYTICS
==========================
Generates high-level academic graphs for project reports.
Includes ROC Curves, PCA, Signal Envelopes, and Tensor Heatmaps.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Import your model
from train_base_model import EMGGestureNet, DEVICE, SAMPLES_PER_REP

# ================= CONFIG =================
MODEL_PATH = "models/base_model_final.pth" 
CONFIG_PATH = "models/model_config.pkl"
DATA_DIR = "data_raw"
OUTPUT_DIR = "report_graphs_advanced"
SAMPLING_RATE = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

def load_config():
    with open(CONFIG_PATH, "rb") as f: return pickle.load(f)

def get_all_data(config):
    """Loads dataset into memory."""
    all_X = []
    all_y = []
    gestures = config['gestures']
    print("📂 Loading full dataset...")
    
    for idx, gesture in enumerate(gestures):
        for root, _, files in os.walk(DATA_DIR):
            if f"{gesture}.csv" in files:
                try:
                    df = pd.read_csv(os.path.join(root, f"{gesture}.csv"))
                    reps = df['rep'].unique()
                    for r in reps:
                        subset = df[df['rep'] == r]
                        if len(subset) >= SAMPLES_PER_REP:
                            d = subset[['adc_ch1', 'adc_ch2']].values[:SAMPLES_PER_REP].T
                            # Normalize
                            d = (d - np.mean(d)) / (np.std(d) + 1e-6)
                            all_X.append(d)
                            all_y.append(idx)
                except: pass
    return np.array(all_X), np.array(all_y), gestures

# --- GRAPH 1: ROC CURVES ---
def plot_roc_curves(model, X, y, gestures):
    print("📈 Generating ROC Curves...")
    
    # Binarize the output (One-vs-Rest)
    y_bin = label_binarize(y, classes=range(len(gestures)))
    n_classes = y_bin.shape[1]
    
    # Get Model Probabilities
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    y_score = []
    model.eval()
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs[0].to(DEVICE)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            y_score.extend(probs.cpu().numpy())
    
    y_score = np.array(y_score)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", n_classes)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{gestures[i].upper()} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, "A_ROC_Curves.png"), dpi=150)
    plt.close()

# --- GRAPH 2: SIGNAL ENVELOPE ---
def plot_signal_envelope(X, y, gestures):
    print("〰️ Generating Signal Envelopes...")
    # Find a sample of "Strong Fist"
    fist_idx = gestures.index("strong_fist") if "strong_fist" in gestures else 0
    sample_idx = np.where(y == fist_idx)[0][0]
    raw_sig = X[sample_idx][0] # Channel 1
    
    # Rectify (Absolute Value)
    rectified = np.abs(raw_sig)
    # Envelope (Moving Average)
    window = 50
    envelope = pd.Series(rectified).rolling(window=window).mean()
    
    plt.figure(figsize=(12, 5))
    plt.plot(raw_sig, label='Raw EMG', alpha=0.5, color='gray')
    plt.plot(rectified, label='Rectified', alpha=0.3, color='orange')
    plt.plot(envelope, label=f'Envelope (MovAvg {window})', color='blue', linewidth=2)
    plt.title(f"EMG Preprocessing: Envelope Extraction ({gestures[fist_idx]})")
    plt.xlabel("Time Samples")
    plt.ylabel("Amplitude (Normalized)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "B_Signal_Envelope.png"), dpi=150)
    plt.close()

# --- GRAPH 3: CNN INPUT HEATMAP ---
def plot_input_heatmap(X, y, gestures):
    print("🔥 Generating Tensor Heatmap...")
    # Plot one sample for each gesture
    fig, axes = plt.subplots(len(gestures), 1, figsize=(10, 2*len(gestures)))
    
    for i, gesture in enumerate(gestures):
        idx = np.where(y == i)[0]
        if len(idx) > 0:
            sample = X[idx[0]] # (2, 600)
            sns.heatmap(sample, ax=axes[i], cmap="viridis", cbar=False)
            axes[i].set_title(f"CNN Input Tensor: {gesture.upper()}")
            axes[i].set_ylabel("Channels")
            axes[i].set_yticks([0.5, 1.5])
            axes[i].set_yticklabels(["CH1", "CH2"])
            if i < len(gestures)-1: axes[i].set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "C_Tensor_Heatmap.png"), dpi=150)
    plt.close()

# --- GRAPH 4: PCA PROJECTION ---
def plot_pca(X, y, gestures):
    print("📐 Generating PCA Projection...")
    # Flatten: (N, 2, 600) -> (N, 1200)
    X_flat = X.reshape(X.shape[0], -1)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    
    plt.figure(figsize=(10, 8))
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Gesture'] = [gestures[i] for i in y]
    
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Gesture', palette='bright', s=80, alpha=0.7)
    plt.title(f'PCA Projection (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.savefig(os.path.join(OUTPUT_DIR, "D_PCA_Clusters.png"), dpi=150)
    plt.close()

# --- GRAPH 5: CLASS BALANCE ---
def plot_class_balance(y, gestures):
    print("📊 Generating Class Balance Chart...")
    counts = [np.sum(y == i) for i in range(len(gestures))]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=gestures, y=counts, palette="viridis")
    plt.title("Dataset Class Distribution")
    plt.ylabel("Number of Samples")
    plt.savefig(os.path.join(OUTPUT_DIR, "E_Class_Balance.png"), dpi=150)
    plt.close()

def main():
    print("🚀 STARTING ADVANCED ANALYTICS...")
    config = load_config()
    
    # Load Model
    model = EMGGestureNet(num_gestures=len(config['gestures']), use_features=config['use_features'])
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
    except:
        print("❌ Model not found. Run training first.")
        return

    # Load Data
    X, y, gestures = get_all_data(config)
    if len(X) == 0:
        print("❌ No data found.")
        return

    # Generate Graphs
    plot_roc_curves(model, X, y, gestures)
    plot_signal_envelope(X, y, gestures)
    plot_input_heatmap(X, y, gestures)
    plot_pca(X, y, gestures)
    plot_class_balance(y, gestures)
    
    print(f"\n✅ All graphs saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()