import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= CONFIGURATION =================
INPUT_DIR = "data_raw"       # Where your CSV files are
OUTPUT_DIR = "graphs_final"  # Where to save the images
DPI = 300                    # High resolution for presentations (standard is 72-100)
SHOW_REPETITIONS = True      # Set to True to draw vertical lines for each rep
# =================================================

def process_file(filename):
    csv_path = os.path.join(INPUT_DIR, filename)
    
    # 1. Read the Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return

    # Check if empty
    if df.empty:
        print(f"⚠️ Skipped {filename} (empty file)")
        return

    gesture_name = filename.replace(".csv", "").replace("_", " ").title()
    total_samples = len(df)
    duration_seconds = df['ms'].iloc[-1] / 1000
    avg_hz = total_samples / duration_seconds if duration_seconds > 0 else 0

    print(f"Processing: {gesture_name} ({avg_hz:.1f} Hz)...")

    # 2. Setup the Plot (High Quality)
    plt.style.use('seaborn-v0_8-whitegrid') # Cleaner look for papers/presentations
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    
    # Title with technical details
    fig.suptitle(f"EMG Signal: {gesture_name}\n{total_samples} samples @ ~{int(avg_hz)} Hz", fontsize=16, fontweight='bold')

    # --- Convert ms to seconds for easier reading ---
    time_sec = df['ms'] / 1000

    # 3. Plot Sensor 1
    ax1.plot(time_sec, df['adc_ch1'], color="#1f77b4", linewidth=0.8, label="Sensor 1 (Pin 34)")
    ax1.set_ylabel("Amplitude (ADC)", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.set_title("Muscle Group 1", fontsize=12)

    # 4. Plot Sensor 2
    ax2.plot(time_sec, df['adc_ch2'], color="#ff7f0e", linewidth=0.8, label="Sensor 2 (Pin 35)")
    ax2.set_xlabel("Time (seconds)", fontsize=14)
    ax2.set_ylabel("Amplitude (ADC)", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.set_title("Muscle Group 2", fontsize=12)

    # 5. (Optional) Draw Repetition Lines
    if SHOW_REPETITIONS and 'rep' in df.columns:
        # Find where the repetition number changes
        rep_changes = df[df['rep'].diff() != 0].index
        
        for i in rep_changes:
            if i == 0: continue # Skip the very first start
            t = time_sec.iloc[i]
            # Draw a faint vertical line on both plots
            ax1.axvline(x=t, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax2.axvline(x=t, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # Label the rep number on the top plot
            rep_num = df['rep'].iloc[i]
            ax1.text(t + 0.1, ax1.get_ylim()[1]*0.9, f"R{rep_num}", fontsize=8, color='gray', rotation=90)

    plt.tight_layout()

    # 6. Save Image
    output_filename = filename.replace(".csv", ".png")
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=DPI)
    plt.close(fig) # Close to free memory
    print(f"   ✅ Saved: {output_path}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Directory '{INPUT_DIR}' not found.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    
    if not files:
        print("No CSV files found.")
        return

    print(f"Found {len(files)} recordings. Generating graphs...")
    for f in files:
        process_file(f)
    
    print("\n🎉 All graphs generated in 'graphs_final/'")

if __name__ == "__main__":
    main()