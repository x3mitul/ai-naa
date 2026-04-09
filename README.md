<div align="center">
  <br />
  <h1>🪞 Ainaa (Aether Mirror Pro v3.5)</h1>
  <p><strong>Reflecting the Best Version of You</strong></p>
  <p>An AI-powered Smart Mirror that tracks posture, profiles styling traits, and enables touch-free interaction using high-performance Edge AI and Computer Vision.</p>
</div>

---

## 🌟 Overview

**Ainaa** is a production-ready "software mirror" designed for high-end digital mirror hardware. It leverages **100% Local Inference** for its core computer vision pipeline, ensuring zero latency and absolute user privacy.

### 🚀 Pro v3.5 Features
- **Studio Mode (New)**: Uses MediaPipe Selfie Segmentation to replace messy room backgrounds with a sleek, dark studio gradient in real-time.
- **Gesture Control (New)**: Interact touch-free! Simply show an "Open Palm" gesture to Ainaa to trigger an instant outfit and style analysis.
- **Biomechanical Posture Engine**: Tracks 33 skeletal landmarks to monitor neck inclination, shoulder tilt, and spine alignment with a live 0-100 composite score.
- **Authentic Styling Profiler**: No harder-coded mocks. Ainaa now uses OpenCV heuristics to sample real frame pixels for detection of:
  - **Skin Tone** (Fair/Olive/Deep)
  - **Face Shape** (Round/Oval/Square based on aspect ratios)
  - **Hairstyle & Hair Color**
  - **Spectacles & Styling Suggestions**
- **Voice-First HUD**: Triggered by "Hey", powered by Web Speech API with real-time feedback and glassmorphic HUD overlays.
- **Local Vault**: SQLite data store persisting your posture and styling journey.


---

## 🏗️ Architecture

Ainaa uses a distributed dual-layer architecture:

- **Frontend (`/src`)**: A lightweight React + Vite application. It renders the HUD, handles the Web Speech API voice capture, and consumes MJPEG frames and WebSocket data from the backend.
- **Backend (`/backend`)**: A heavy-lifting Python FastAPI server. It runs OpenCV camera capture, MediaPipe inference on an isolated background thread, and serves metrics over Http/WebSockets.

---

## ⚡ Getting Started (Local Development)

### Prerequisites
- Node.js (v18+)
- Python 3.10+
- A working webcam

### 1. Start the Brain (Backend)
The backend manages the camera feed and AI inference.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # (or `venv\Scripts\activate` on Windows)

# Install requirements (Assuming you have generated one, else see code)
pip install fastapi uvicorn mediapipe opencv-python numpy google-genai

# Run the server
python main.py
```
> **Note**: On the first run, the backend will automatically download the heavy MediaPipe pose model (~9MB).

### 2. Start the Interface (Frontend)
The frontend serves the smart mirror HUD.

```bash
# From the project root
npm install
npm run dev
```

### 3. Open the Mirror
Navigate to `http://localhost:5173` in **Google Chrome** (Chrome is required for the Web Speech API).
Ensure you allow microphone and camera access.

---

## 🎙️ Voice Commands

Once the interface loads, click **"TAP TO ENABLE VOICE"** (a browser requirement). Then, use the wake word **"Hey"** followed by a command:

- *"Hey... rate my outfit"*
- *"Hey... check my posture"*
- *"Hey... what is my score"*
- *"Hey... what time is it"*

*(If you have a Gemini API key connected, the mirror can also engage in general contextual conversation about your appearance and wellness).*

---

## 🧠 Smart Fallback System (Gemini)

By default, outfit analysis and chat run locally using sophisticated OpenCV segmentation and contextual rule engines.

To unlock **Vision AI**, supply a Google Gemini API Key. The backend will detect it and gracefully upgrade its capabilities:

```bash
export GEMINI_API_KEY="your_api_key_here"
# then run python main.py
```

If your quota is depleted, Ainaa silently falls back to local models without crashing the UI.

---

## 🛠️ Built With
* **Frontend**: React, Vite, Vanilla CSS Design System, WebRTC, Web Speech API
* **Backend**: Python, FastAPI, WebSockets
* **Computer Vision**: OpenCV, MediaPipe Tasks API
* **AI Engine**: Google GenAI SDK (Gemini-2.0-flash), Local OpenCV heuristics
* **Database**: SQLite3

---

<div align="center">
  <p>Built for the Hackathon · Designed for the Future</p>
</div>
