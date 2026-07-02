import serial
import time
import csv
import re
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
PORT = "COM5"        # ⚠️ Check your port
BAUD_RATE = 115200
DURATION = 3.0       # Seconds to record
OUTPUT_FILE = "diagnostic_test.csv"
# ==========================================

def main():
    print(f"🔌 Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2) # Wait for connection to stabilize
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print("\n========================================")
    print("   3-SECOND SIGNAL DIAGNOSTIC")
    print("========================================")
    print("1. Hold your arm ready.")
    print("2. When you press Enter, SQUEEZE A STRONG FIST immediately.")
    print("3. Hold the fist until the recording stops.")
    print("========================================")
    
    input("\n👉 Press Enter to START RECORDING...")
    
    print("\n🔴 RECORDING... SQUEEZE NOW! ✊")
    
    start_time = time.time()
    data = []
    pattern = re.compile(r"(\d+)\D+(\d+)")
    
    # --- RECORDING LOOP ---
    while (time.time() - start_time) < DURATION:
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                match = pattern.search(line)
                if match:
                    c1 = int(match.group(1))
                    c2 = int(match.group(2))
                    data.append([c1, c2])
            except:
                pass
    # ----------------------

    ser.close()
    print("✅ Recording Complete.")
    
    if len(data) == 0:
        print("❌ No data collected. Check connections/baud rate.")
        return

    # --- SAVE TO CSV ---
    try:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["CH1", "CH2"])
            writer.writerows(data)
        print(f"💾 Data saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"⚠️ Could not save CSV: {e}")

    # --- ANALYZE ---
    arr = np.array(data)
    max_c1 = np.max(arr[:, 0])
    max_c2 = np.max(arr[:, 1])
    mean_c1 = np.mean(arr[:, 0])
    mean_c2 = np.mean(arr[:, 1])
    
    print("\n📊 DIAGNOSTIC REPORT")
    print("-" * 30)
    print(f"Samples Collected: {len(data)} ({(len(data)/DURATION):.1f} Hz)")
    print("-" * 30)
    
    # Channel 1 Analysis
    status1 = "✅ GOOD" if max_c1 > 1500 else "⚠️ WEAK"
    if max_c1 > 4090: status1 = "⚠️ CLIPPING (Too High)"
    print(f"Channel 1 Peak: {max_c1}  | Mean: {mean_c1:.1f} -> {status1}")

    # Channel 2 Analysis
    status2 = "✅ GOOD" if max_c2 > 1500 else "⚠️ WEAK"
    if max_c2 > 4090: status2 = "⚠️ CLIPPING (Too High)"
    print(f"Channel 2 Peak: {max_c2}  | Mean: {mean_c2:.1f} -> {status2}")
    print("-" * 30)

    if max_c1 < 500 and max_c2 < 500:
        print("\n❌ FAIL: Signals are too weak compared to training data.")
        print("   Action: Wet the electrodes or check Ground wire.")
    elif max_c1 > 1500 or max_c2 > 1500:
        print("\n✅ PASS: Signal strength matches training data!")
        print("   Action: You are ready to run 'calibrate_and_benchmark_v3.py'")

if __name__ == "__main__":
    main()