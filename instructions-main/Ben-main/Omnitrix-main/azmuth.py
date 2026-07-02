"""
EMG recording script for two sensors on ESP32 (pins 34 & 35) via serial (COM5 by default).

Features implemented per user request:
- Records each repetition for 3 seconds, repeated 15 times (total 45s per gesture).
- GESTURES mapping as requested.
- Prints clear terminal instructions during recording.
- Saves combined data for each gesture into CSV files under `data_raw/`.

Notes:
- Requires pyserial, pandas. Install with: pip install pyserial pandas
- Set PORT and BAUD to match your ESP32 settings.

Usage: run the script and follow the terminal prompts.
"""

import os
import time
import re
from collections import deque
import serial
import pandas as pd

# =========================
# CONFIGURATION
# =========================
PORT = "COM5"          # ⚠️ Change this to your ESP32 port if different
BAUD = 115200
SAMPLING_RATE = 200    # Hz (samples per second) - Updated for higher resolution EMG
DURATION_PER_REP = 3     # seconds per repetition (user requested 3s)
REPETITIONS = 15         # repetitions per gesture (user requested 15)
TOTAL_DURATION = DURATION_PER_REP * REPETITIONS  # 45 seconds total
SAMPLES_PER_REP = int(SAMPLING_RATE * DURATION_PER_REP)  # ~600 samples per repetition
EXPECTED_SAMPLES_TOTAL = int(SAMPLING_RATE * TOTAL_DURATION)  # ~9000 samples total
SAVE_DIR = "data_raw"
GRAPH_DIR = "graphs"

GESTURES = {
    "rest": "Neutral / no message",
    "fist": "Yes / OK",
    "open": "No / Stop",
    "wrist_up": "Hello / Start speaking",
    "wrist_down": "Goodbye / End speaking",
    "strong_fist": "Urgent / Need help"
}

# How frequently to update the progress print (seconds).
PLOT_UPDATE_INTERVAL = 0.1  # kept as a small interval marker (not used for plotting)


# =========================
# HELPERS
# =========================

def parse_dual_channel(line):
    """
    Extract two integers from a serial line.
    Accepts formats like: '512,478' or 'EMG1:512 EMG2:478' or '512 478'.
    Returns tuple (int, int) or (None, None) if parsing fails.
    """
    nums = re.findall(r"\d+", line)
    if len(nums) >= 2:
        try:
            return int(nums[0]), int(nums[1])
        except ValueError:
            return None, None
    return None, None


def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    # graphs directory removed from automatic creation since plotting was removed
    # os.makedirs(GRAPH_DIR, exist_ok=True)


def adc_to_volt(adc_value, vref=3.3, resolution=4095):
    """Convert ADC integer (0..resolution) to voltage (V)."""
    try:
        return (adc_value / resolution) * vref
    except Exception:
        return 0.0


def record_gesture(ser, gesture_name):
    """
    Record one gesture for REPETITIONS × DURATION_PER_REP seconds at SAMPLING_RATE Hz.

    This function performs:
    - Prompts for the user and then records continuously for TOTAL_DURATION.
    - Collects data into per-gesture buffers (list of rows).
    - Saves combined CSV for the gesture.

    Plotting and live graphs were removed.
    """

    print(f"\n=== Gesture: {gesture_name.upper()} ===")
    print(f"Meaning: {GESTURES.get(gesture_name, 'N/A')}")
    print(f"📊 Expected data points: ~{EXPECTED_SAMPLES_TOTAL} samples at {SAMPLING_RATE}Hz")

    # Prepare buffers that will hold the entire 45s of data for this gesture
    timestamps = deque()  # seconds since start of first rep for this gesture
    emg1 = deque()
    emg2 = deque()
    rows = []  # list to store CSV rows (ms, adc_ch1, volt_ch1, adc_ch2, volt_ch2, rep)

    overall_start = None

    try:
        # Single prompt to start continuous recording for all repetitions
        input(f"\n👉 Prepare for '{gesture_name}'. Press Enter to start continuous recording ({REPETITIONS} reps × {DURATION_PER_REP}s = {TOTAL_DURATION}s @ {SAMPLING_RATE}Hz)...")
        print(f"Starting continuous recording for gesture '{gesture_name}' ({TOTAL_DURATION}s total)...")
        ser.reset_input_buffer()
        overall_start = time.time()

        last_print_time = 0
        current_rep = 1
        samples_collected = 0
        print(f"Recording repetition {current_rep} of {REPETITIONS} ({DURATION_PER_REP}s)...")

        # Loop until the total duration for this gesture elapses
        # Loop until the total duration for this gesture elapses
        while (time.time() - overall_start) < TOTAL_DURATION:
            try:
                # This is a BLOCKING READ.
                # It will wait for the ESP32 to send a full line (ending in '\n').
                # This is the fastest, most efficient way.
                raw_line = ser.readline().decode(errors="ignore").strip()
                
                # If we get a timeout (0.1s) or an empty line, just loop again
                if not raw_line:
                    continue 
                
                # --- The rest of your code is the same ---
                
                ch1, ch2 = parse_dual_channel(raw_line)
                if ch1 is None or ch2 is None:
                    # Could not parse this line; skip
                    continue

                t = time.time() - overall_start
                # Determine which repetition this timestamp belongs to (1-based)
                rep_idx = int(t // DURATION_PER_REP) + 1
                if rep_idx > REPETITIONS:
                    rep_idx = REPETITIONS

                # If we've entered a new repetition, notify the user
                if rep_idx != current_rep:
                    current_rep = rep_idx
                    print(f"Recording repetition {current_rep} of {REPETITIONS} ({DURATION_PER_REP}s)... [{samples_collected} samples so far]")

                timestamps.append(t)
                emg1.append(ch1)
                emg2.append(ch2)
                samples_collected += 1

                ms = int((t) * 1000)
                rows.append([ms, ch1, adc_to_volt(ch1), ch2, adc_to_volt(ch2), rep_idx])

                # NO time.sleep(0.001) HERE

            except Exception as e:
                print("⚠️ Error while reading/parsing serial data:", e)
                # continue collecting if possible

        print(f"\nDone. Continuous recording finished for this gesture. Collected {samples_collected} samples.")
        print("Recording complete! Saving all data...")
        ensure_dirs()

        # Save CSV for the gesture
        if len(rows) == 0:
            print("⚠️ No data was captured for this gesture.")
        else:
            df = pd.DataFrame(rows, columns=["ms", "adc_ch1", "volt_ch1", "adc_ch2", "volt_ch2", "rep"])
            save_path = os.path.join(SAVE_DIR, f"{gesture_name}.csv")
            df.to_csv(save_path, index=False)
            print(f"✅ Saving data to: {save_path} ({len(df)} rows)")
            print(f"   Sampling rate achieved: {len(df) / TOTAL_DURATION:.1f} Hz")

    finally:
        # No plotting resources to close
        pass


def main():
    print("🔌 Connecting to ESP32...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        # Small delay to let the serial connection initialize
        time.sleep(2)
        print(f"✅ Connected to {PORT} at {BAUD} baud.")
    except Exception as e:
        print(f"❌ Could not open serial port {PORT}: {e}")
        return

    try:
        for gesture in GESTURES:
            record_gesture(ser, gesture)

        print("\n🎉 All recordings complete! Files saved in 'data_raw/'.\n")

    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

