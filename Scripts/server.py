import os
import re
import time
import serial
import torch
import numpy as np
import pickle
import collections
import threading
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from train_base_model import EMGGestureNet, DEVICE

# ================= CONFIGURATION =================
SERIAL_PORT = "COM5"   # ⚠️ CHECK YOUR PORT
BAUD_RATE = 115200
MODEL_DIR = "models"
ADAPTED_MODELS_DIR = os.path.join(MODEL_DIR, "adapted_models")
USER_ID = "01"         # Set to None if using base model

# Global variables
latest_prediction = {
    "gesture": "WAITING...",
    "confidence": 0.0,
    "locked": False
}

app = FastAPI()

# ================= THE WEBSITE (HTML/CSS/JS) =================
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gesture Communication Board</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #1a1a1a;
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            transition: background-color 0.4s ease;
        }

        .container {
            text-align: center;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            padding: 50px;
            border-radius: 30px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.6);
            width: 80%;
            max-width: 500px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        h2 { 
            color: rgba(255,255,255,0.6); 
            font-weight: 300; 
            margin: 0 0 10px 0; 
            font-size: 1rem;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        
        /* The main gesture name (e.g., FIST) */
        #gesture-display {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
            opacity: 0.5;
            margin-bottom: 20px;
        }

        /* The big action message (e.g., NO, HELLO) */
        #action-message {
            font-size: 4rem;
            margin-bottom: 30px;
            font-weight: 800;
            text-shadow: 0 4px 10px rgba(0,0,0,0.3);
            min-height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: popIn 0.3s ease-out;
        }

        @keyframes popIn {
            0% { transform: scale(0.8); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 20px;
        }
        #confidence-fill {
            height: 100%;
            width: 0%;
            background-color: #fff;
            transition: width 0.1s linear, background-color 0.3s ease;
        }
        .stats {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 0.8rem;
            color: rgba(255,255,255,0.5);
        }
        
        .status-badge {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 8px 15px;
            border-radius: 20px;
            background: rgba(0,0,0,0.5);
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .dot { height: 8px; width: 8px; border-radius: 50%; background: #ff4444; }
    </style>
</head>
<body>
    <div class="status-badge">
        <div class="dot" id="status-dot"></div>
        <span id="status-text">Disconnected</span>
    </div>

    <div class="container">
        <h2>Detected Gesture</h2>
        <div id="gesture-display">...</div>
        
        <div id="action-message">💤 None</div>
        
        <div class="progress-bar">
            <div id="confidence-fill"></div>
        </div>
        
        <div class="stats">
            <span>Confidence</span>
            <span id="conf-value">0%</span>
        </div>
    </div>

    <script>
        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        const wsUrl = `${protocol}://${window.location.host}/ws`;
        const ws = new WebSocket(wsUrl);
        
        const display = document.getElementById("gesture-display");
        const message = document.getElementById("action-message");
        const confFill = document.getElementById("confidence-fill");
        const confValue = document.getElementById("conf-value");
        const statusDot = document.getElementById("status-dot");
        const statusText = document.getElementById("status-text");
        const body = document.body;

        let currentGesture = "";

        ws.onopen = () => {
            statusDot.style.background = "#00ff88";
            statusText.innerText = "System Online";
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Only update animation if gesture changed
            if (data.gesture !== currentGesture) {
                message.style.animation = 'none';
                message.offsetHeight; /* trigger reflow */
                message.style.animation = 'popIn 0.3s ease-out';
                currentGesture = data.gesture;
            }

            display.innerText = data.gesture;
            confValue.innerText = data.confidence + "%";
            confFill.style.width = data.confidence + "%";

            // --- CUSTOM MESSAGES & COLORS ---
            let bgColor = "#1a1a1a";
            let textColor = "#fff";
            let msgText = "💤 None";

            if (data.gesture === "STRONG_FIST") {
                bgColor = "#4a0000";      // Deep Red
                textColor = "#ff6b6b";    // Bright Red Text
                msgText = "❌ No";
                
            } else if (data.gesture === "OPEN") {
                bgColor = "#004a2f";      // Deep Green
                textColor = "#5cffa9";    // Bright Green Text
                msgText = "👋 Hello";
                
            } else if (data.gesture === "WRIST_UP") {
                bgColor = "#002a4a";      // Deep Blue
                textColor = "#6bbaff";    // Bright Blue Text
                msgText = "👌 Ok";
                
            } else if (data.gesture === "WRIST_DOWN") {
                bgColor = "#5c4d00";      // Deep Gold
                textColor = "#ffd700";    // Bright Gold Text
                msgText = "🙏 Thank You";
                
            } else if (data.gesture === "REST") {
                bgColor = "#1a1a1a";      // Dark Gray
                textColor = "#888";       // Gray Text
                msgText = "😶 None";
            }

            // Apply Updates
            if (data.gesture !== "WAITING..." && data.gesture !== "SERIAL ERROR") {
                body.style.backgroundColor = bgColor;
                message.style.color = textColor;
                message.innerText = msgText;
                confFill.style.backgroundColor = textColor;
            }
        };

        ws.onclose = () => {
            statusDot.style.background = "#ff4444";
            statusText.innerText = "Offline";
        };
    </script>
</body>
</html>
"""

# ================= HELPER CLASSES =================
class DataBuffer:
    def __init__(self, channels=2, max_len=600):
        self.max_len = max_len
        self.buffer = collections.deque(maxlen=max_len)
    def add(self, sample): self.buffer.append(sample)
    def is_ready(self): return len(self.buffer) == self.max_len
    def get_window(self): return np.array(self.buffer).T

class StabilityFilter:
    def __init__(self, required_frames=4):
        self.required_frames = required_frames
        self.current_display = "REST"
        self.pending_gesture = None
        self.counter = 0

    def update(self, new_gesture, confidence):
        if confidence < 0.70: return self.current_display, False
        if new_gesture == self.pending_gesture:
            self.counter += 1
        else:
            self.pending_gesture = new_gesture
            self.counter = 1
            
        if self.counter >= self.required_frames:
            self.current_display = new_gesture
            self.counter = self.required_frames
            return self.current_display, True
        return self.current_display, False

# ================= LOGIC =================
def run_inference_loop():
    global latest_prediction
    try:
        with open(os.path.join(MODEL_DIR, "model_config.pkl"), "rb") as f: config = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "gesture_mappings.pkl"), "rb") as f: mappings = pickle.load(f)
    except: return

    idx_to_gesture = mappings['idx_to_gesture']
    
    # Load Model
    model = EMGGestureNet(num_gestures=len(config['gestures']), use_features=config['use_features'])
    user_path = os.path.join(ADAPTED_MODELS_DIR, f"user_{USER_ID}_model.pth")
    model_path = user_path if os.path.exists(user_path) else os.path.join(MODEL_DIR, "base_model_final.pth")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    # Load Norm
    norm_path = os.path.join(ADAPTED_MODELS_DIR, f"user_{USER_ID}_normalization.pkl")
    if not os.path.exists(norm_path): norm_path = os.path.join(MODEL_DIR, "normalization_params.pkl")
    with open(norm_path, "rb") as f: norm_params = pickle.load(f)

    # Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"✅ Serial Connected")
    except Exception as e:
        print(f"❌ Serial Error: {e}")
        latest_prediction["gesture"] = "SERIAL ERROR"
        return

    buffer = DataBuffer(max_len=config['samples_per_rep'])
    stabilizer = StabilityFilter()
    digit_pattern = re.compile(r"(\d+)\D+(\d+)")
    
    print("🚀 Engine Running...")
    
    while True:
        if ser.in_waiting:
            lines = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').split('\n')
            for line in lines:
                match = digit_pattern.search(line)
                if match: buffer.add([int(match.group(1)), int(match.group(2))])
        
        if buffer.is_ready():
            window = buffer.get_window()
            # Normalize
            if 'mean_ch1' in norm_params:
                m1, s1 = norm_params['mean_ch1'], norm_params['std_ch1']
                m2, s2 = norm_params['mean_ch2'], norm_params['std_ch2']
            else:
                m1, s1 = norm_params['global_mean_ch1'], norm_params['global_std_ch1']
                m2, s2 = norm_params['global_mean_ch2'], norm_params['global_std_ch2']
            
            norm = np.zeros_like(window, dtype=float)
            norm[0,:] = (window[0,:] - m1)/(s1+1e-8)
            norm[1,:] = (window[1,:] - m2)/(s2+1e-8)
            
            # Inference
            inp = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(model(inp), dim=1)
                conf, pred = torch.max(probs, 1)
            
            final_gesture, _ = stabilizer.update(idx_to_gesture[pred.item()], conf.item())
            
            latest_prediction = {
                "gesture": final_gesture.upper(),
                "confidence": round(conf.item() * 100, 1)
            }
            time.sleep(0.05)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_inference_loop, daemon=True).start()

# ================= ROUTES =================
@app.get("/")
async def get():
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(latest_prediction)
            await asyncio.sleep(0.05)
    except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)