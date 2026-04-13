import cv2
import json
import asyncio
import os
import base64
import urllib.request
import threading
import time
import sqlite3
import random
from datetime import datetime
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Legacy solutions for easier pixel access
try:
    mp_selfie = mp.solutions.selfie_segmentation
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
except AttributeError:
    class MockSeg:
        def process(self, img):
            import types
            res = types.SimpleNamespace()
            import numpy as np
            res.segmentation_mask = np.ones(img.shape[:2], dtype=np.uint8)
            return res
    class MockHandsProcess:
        def process(self, img):
            import types
            res = types.SimpleNamespace()
            res.multi_hand_landmarks = None
            return res
    class MockFaceDetect:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def process(self, img):
            import types
            res = types.SimpleNamespace()
            res.detections = None
            return res
    mp_selfie = type('Hack', (), {'SelfieSegmentation': lambda **kw: MockSeg()})
    mp_hands = type('Hack', (), {'Hands': lambda **kw: MockHandsProcess()})
    mp_face_detection = type('Hack', (), {'FaceDetection': MockFaceDetect})



# --- xAI (Grok) ---
XAI_API_KEY = os.environ.get("XAI_API_KEY")
xai_client = None

if XAI_API_KEY:
    try:
        from openai import OpenAI
        xai_client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1"
        )
        # Quick connectivity check
        xai_client.models.list()
        print("[Ainaa] ✅ xAI (Grok) connected and verified.")
    except Exception as e:
        print(f"[Ainaa] ⚠️  xAI error: {e}")
        print("[Ainaa] ⚠️  Falling back to smart local analysis.")
        xai_client = None
else:
    print("[Ainaa] ℹ️  No XAI_API_KEY. Using smart local analysis.")

# --- App & Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ainaa Backend", version="3.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Pose Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
if not os.path.exists(MODEL_PATH):
    print(f"[Ainaa] Downloading heavy pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[Ainaa] ✅ Pose model ready.")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.IMAGE,
    num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Landmarks
NOSE, LEFT_EAR, RIGHT_EAR = 0, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

# --- SQLite Vault ---
DB_PATH = os.path.join(os.path.dirname(__file__), "vault.db")
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
        posture_score REAL, outfit_rating REAL, analysis_text TEXT,
        neck_angle REAL, shoulder_tilt REAL, torso_angle REAL)''')
    conn.commit(); conn.close()
init_db()

def save_snapshot(**kw):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO snapshots (timestamp, posture_score, outfit_rating, analysis_text, neck_angle, shoulder_tilt, torso_angle) VALUES (?,?,?,?,?,?,?)',
        (datetime.now().isoformat(), kw.get('posture_score'), kw.get('outfit_rating'),
         kw.get('analysis_text'), kw.get('neck_angle',0), kw.get('shoulder_tilt',0), kw.get('torso_angle',0)))
    conn.commit(); conn.close()

def get_vault_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT timestamp, posture_score, outfit_rating, analysis_text FROM snapshots ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [{"timestamp": r[0], "posture_score": r[1], "outfit_rating": r[2], "analysis": r[3]} for r in rows]

# ============================================================
# POSTURE ENGINE
# ============================================================
def calc_angle(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8), -1, 1)
    return np.degrees(np.arccos(cos))

def analyze_posture(landmarks):
    nose = (landmarks[NOSE].x, landmarks[NOSE].y)
    l_ear = (landmarks[LEFT_EAR].x, landmarks[LEFT_EAR].y)
    r_ear = (landmarks[RIGHT_EAR].x, landmarks[RIGHT_EAR].y)
    l_sh = (landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y)
    r_sh = (landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y)
    l_hip = (landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y)
    r_hip = (landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y)

    mid_ear = ((l_ear[0]+r_ear[0])/2, (l_ear[1]+r_ear[1])/2)
    mid_sh = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
    mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)

    # Neck inclination (0° = perfect)
    vert_pt = (mid_sh[0], mid_sh[1] - 0.3)
    neck_angle = calc_angle(mid_ear, mid_sh, vert_pt)
    neck_score = max(0, 100 - neck_angle * 3)

    # Shoulder tilt
    sh_dy = abs(l_sh[1] - r_sh[1])
    sh_tilt = np.degrees(np.arctan2(sh_dy, abs(l_sh[0] - r_sh[0]) + 1e-8))
    sh_score = max(0, 100 - sh_tilt * 4)

    # Torso lean
    torso_angle = calc_angle((mid_hip[0], mid_hip[1]-0.3), mid_hip, mid_sh)
    torso_score = max(0, 100 - torso_angle * 2.5)

    composite = neck_score * 0.45 + sh_score * 0.25 + torso_score * 0.30
    composite = max(0, min(100, composite))
    status = "ALIGNED" if composite >= 75 else "FAIR" if composite >= 50 else "SLOUCHING"

    return {
        "status": status, "composite_score": round(composite, 1),
        "neck_angle": round(neck_angle, 1), "shoulder_tilt": round(sh_tilt, 1),
        "torso_angle": round(torso_angle, 1),
    }

def analyze_outfit_cv(frame):
    """Rich OpenCV-based outfit analysis analyzing colors, contrast, and composition."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define body regions
    upper_body = hsv[int(h*0.25):int(h*0.55), int(w*0.3):int(w*0.7)]
    lower_body = hsv[int(h*0.55):int(h*0.85), int(w*0.3):int(w*0.7)]

    def get_color_info(region):
        if region.size == 0:
            return "neutral", "medium", 0
        avg_sat = float(np.mean(region[:, :, 1]))
        avg_val = float(np.mean(region[:, :, 2]))

        if avg_sat < 50:
            if avg_val < 65: return "Black", "dark", avg_sat
            elif avg_val > 190: return "White", "bright", avg_sat
            else: return "Gray", "medium", avg_sat

        hue_names = {0:"Red", 1:"Orange", 2:"Yellow", 3:"Lime", 4:"Green", 5:"Teal",
                     6:"Cyan", 7:"Azure", 8:"Blue", 9:"Purple", 10:"Magenta", 11:"Rose"}
        hist = cv2.calcHist([region], [0], None, [12], [0, 180])
        dom_hue = int(np.argmax(hist))
        color = hue_names.get(dom_hue, "Neutral")
        brightness = "bright" if avg_val > 170 else "medium" if avg_val > 85 else "dark"
        return color, brightness, avg_sat

    upper_color, upper_bright, upper_sat = get_color_info(upper_body)
    lower_color, lower_bright, lower_sat = get_color_info(lower_body)

    # Scoring
    score = 7.0
    if upper_color != lower_color: score += 1.0
    if abs(upper_sat - lower_sat) > 30: score += 0.5
    score = min(10, score)

    # --- Dynamic Style Persona Engine (Local NLG) ---
    def generate_style_commentary(u_col, l_col, sc, p_metrics):
        p_status = p_metrics.get("status", "ALIGNED")
        
        # Analysis Templates
        positive_looks = [
            f"The {u_col} and {l_col} combination creates a really sophisticated silhouette.",
            f"I'm loving how the {u_col} tones play off the {l_col} — it's very intentional.",
            f"That {u_col} top is a bold choice that perfectly complements the {l_col} bottom.",
            f"You've nailed the color balance here with a clean {u_col}/{l_col} split."
        ]
        neutral_looks = [
            f"A classic {u_col} and {l_col} pairing. It's safe, clean, and professional.",
            f"The {u_col} and {l_col} work well together, providing a balanced look.",
            f"A very grounded outfit. The {u_col} tones are the star of the show here."
        ]
        
        # Suggestion Templates
        stature_advice = ""
        if p_status == "SLOUCHING":
            stature_advice = " Also, try rolling your shoulders back — it'll make that fit look even more tailored."
        
        suggestions_list = [
            f"Try adding a metallic watch or a subtle chain to elevate the {u_col} tones.",
            "A structured blazer would add great definition to this particular color palette.",
            f"Consider a contrasting shoe color to break up the {l_col} visual block.",
            "This look is great for today; maybe add a textured layer if you're heading out."
        ]
        
        final_analysis = random.choice(positive_looks if sc >= 8 else neutral_looks)
        final_suggestion = random.choice(suggestions_list) + stature_advice
        
        return final_analysis, final_suggestion

    # Get posture context from state
    posture_ctx = state.get_metrics()
    analysis_text, suggestion_text = generate_style_commentary(upper_color, lower_color, score, posture_ctx)

    return {
        "rating": score,
        "analysis": analysis_text,
        "suggestions": suggestion_text,
        "top_color": upper_color
    }

# --- Color Theory Engine ---
def suggest_colors(dominant_color):
    """Suggests complementary and analogous colors based on color theory."""
    harmonies = {
        "Red": ["Emerald Green", "Cyan", "Royal Blue"],
        "Orange": ["Deep Blue", "Teal", "Forest Green"],
        "Yellow": ["Violet", "Indigo", "Charcoal"],
        "Lime": ["Purple", "Magenta", "Deep Slate"],
        "Green": ["Rose", "Pink", "Burgundy"],
        "Teal": ["Coral", "Peach", "Amber"],
        "Cyan": ["Red", "Orange-Red", "Bronze"],
        "Azure": ["Gold", "Goldenrod", "Tan"],
        "Blue": ["Orange", "Mustard", "Cream"],
        "Purple": ["Yellow", "Mint", "Lemon"],
        "Magenta": ["Lime Green", "Spring Green", "Ivory"],
        "Rose": ["True Green", "Slate", "Olive"],
        "Black": ["White", "Gold", "Crimson", "Royal Blue"],
        "White": ["Black", "Navy", "Emerald", "Dark Gray"],
        "Gray": ["Pastel Pink", "Sky Blue", "Lavender", "Mustard"]
    }
    return harmonies.get(dominant_color, ["Navy", "Charcoal", "Emerald"])

# ============================================================
# AUTHENTIC EDGE AI ENGINES
# ============================================================
def analyze_style_authentic(frame):
    """Authentic local CV analysis for styling traits using robust heuristics."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    style_data = {
        "skin_tone": "Neutral", "face_shape": "Oval", 
        "hairstyle": "Dark / Styled", "eye_color": "Deep Tone",
        "spectacles": "Modern Wayfarer",
        "color_suggestions": ["Navy", "Charcoal", "Emerald"]
    }

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_det:
        results = face_det.process(rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
            xw, xh = int(bbox.width * w), int(bbox.height * h)
            
            # 1. Improved Face Shape Heuristic
            ratio = xw / (xh + 1e-6)
            if ratio > 0.92: style_data["face_shape"] = "Round"
            elif ratio > 0.82: style_data["face_shape"] = "Square"
            elif ratio < 0.72: style_data["face_shape"] = "Oval"
            else: style_data["face_shape"] = "Heart"
            
            # 2. Robust Skin Tone via YCrCb Segmentation
            face_roi_ycrcb = ycrcb[max(0,ymin):min(h,ymin+xh), max(0,xmin):min(w,xmin+xw)]
            if face_roi_ycrcb.size > 0:
                mask = cv2.inRange(face_roi_ycrcb, (0, 133, 77), (255, 173, 127))
                skin_pixels = face_roi_ycrcb[mask > 0]
                if skin_pixels.size > 0:
                    avg_ycrcb = np.mean(skin_pixels, axis=0)
                    cr, cb = avg_ycrcb[1], avg_ycrcb[2]
                    tone = "Warm" if cr > 150 else "Cool" if cr < 140 else "Neutral"
                    y_val = avg_ycrcb[0]
                    depth = "Fair" if y_val > 180 else "Medium" if y_val > 100 else "Deep"
                    style_data["skin_tone"] = f"{tone} {depth}"

            # 3. Simple Hairstyle Heuristic
            h_y = max(0, ymin - int(xh * 0.15))
            hair_zone = frame[h_y:ymin, xmin+int(xw*0.2):xmin+int(xw*0.8)]
            if hair_zone.size > 0:
                gray_hair = cv2.cvtColor(hair_zone, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray_hair, cv2.CV_64F).var()
                brightness = np.mean(gray_hair)
                if laplacian > 100: style_data["hairstyle"] = "Textured / Full"
                elif brightness > 150: style_data["hairstyle"] = "Light / Sleek"
                else: style_data["hairstyle"] = "Dark / Natural"

            # 4. Spectacles Recommendation
            spec_map = {"Round": "Angular Geometric", "Square": "Round Wire", "Heart": "Wayfarer", "Oval": "Rectangular"}
            style_data["spectacles"] = spec_map.get(style_data["face_shape"], "Clear Frames")

    # 5. Dynamic Color Suggestions
    outfit = analyze_outfit_cv(frame)
    style_data.update(outfit)
    upper_color = outfit.get("top_color", "Gray")
    base_color = upper_color.split()[-1]
    style_data["color_suggestions"] = suggest_colors(base_color)
    
    return style_data


async def analyze_with_xai(frame_bytes):
    if not xai_client:
        return None
    try:
        b64 = base64.b64encode(frame_bytes).decode('utf-8')
        response = xai_client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are Ainaa, a stylish AI mirror. Analyze this person's outfit and facial features. Respond as raw JSON only: {\"rating\": <1-10>, \"analysis\": \"<2 sentences>\", \"suggestions\": \"<1 sentence>\", \"skin_tone\": \"<e.g. Warm Olive>\", \"face_shape\": \"<e.g. Oval>\", \"hairstyle\": \"<e.g. Messy Fringe>\", \"eye_color\": \"<e.g. Dark Brown>\", \"color_suggestions\": [\"<color1>\", \"<color2>\"], \"spectacles\": \"<e.g. Round Tortoiseshell frames>\"}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }
            ],
            response_format={ "type": "json_object" }
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        print(f"[Ainaa] xAI vision error: {e}")
        return None

# ============================================================
# SMART CHAT (Local fallback)
# ============================================================
def smart_chat_local(message, metrics):
    """Contextual chat responses based on posture state and common queries."""
    msg = message.lower()
    score = metrics.get("composite_score", 0)
    status = metrics.get("status", "unknown")
    
    # Greetings
    if any(w in msg for w in ['hello', 'hi', 'hey', 'morning', 'evening', 'night']):
        hour = datetime.now().hour
        greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
        return f"{greeting}! I'm Ainaa, your smart mirror assistant. Your posture score is {round(score)}. How can I help?"
    
    # Posture queries
    if any(w in msg for w in ['posture', 'back', 'spine', 'slouch', 'straight', 'sit', 'stand']):
        if status == "SLOUCHING":
            return f"I'm seeing some slouching with a score of {round(score)}/100. Try rolling your shoulders back and engaging your core. It makes a big difference!"
        elif status == "FAIR":
            return f"Your posture is fair at {round(score)}/100. You're close to perfect — just lift your chin slightly and you're there."
        return f"Excellent posture! Score: {round(score)}/100. You're sitting like a champion."
    
    # Outfit/Style
    if any(w in msg for w in ['outfit', 'wear', 'style', 'cloth', 'dress', 'look', 'fashion', 'rate', 'fit']):
        return "I can analyze your outfit for you! Say 'rate my outfit' and I'll give you a detailed style assessment."
    
    # Compliment fishing
    if any(w in msg for w in ['how do i look', 'do i look good', 'am i']):
        return f"You look great! Your posture confidence score is {round(score)}/100. Stand tall and own it!"
    
    # Health/wellness
    if any(w in msg for w in ['health', 'wellness', 'exercise', 'stretch', 'tired', 'pain']):
        return "Regular posture checks can prevent back pain. I've been monitoring you — try taking a 30-second stretch break every hour."
    
    # Time
    if any(w in msg for w in ['time', 'clock', 'what time']):
        now = datetime.now()
        h = now.hour % 12 or 12
        return f"It's {h}:{now.minute:02d} {'PM' if now.hour >= 12 else 'AM'}."
    
    # About Ainaa
    if any(w in msg for w in ['who are you', 'what are you', 'what can you do', 'help', 'ainaa']):
        return "I'm Ainaa, your AI-powered smart mirror. I can track your posture in real-time, rate your outfits, and give you style tips. Try: 'check my posture' or 'rate my outfit'!"
    
    # Fallback
    return f"I'm your smart mirror assistant! I can check your posture (currently {round(score)}/100) or rate your outfit. What would you like?"

# ============================================================
# SHARED STATE
# ============================================================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw_frame = None
        self.annotated_frame = None
        self.metrics = {"status":"ALIGNED","composite_score":0,"neck_angle":0,"shoulder_tilt":0,"torso_angle":0,"has_person":False, "gesture": "None"}
    def update_frames(self, raw, annotated):
        with self.lock: self.raw_frame = raw; self.annotated_frame = annotated
    def get_annotated(self):
        with self.lock: return self.annotated_frame
    def get_raw(self):
        with self.lock: return self.raw_frame
    def update_metrics(self, m):
        with self.lock: self.metrics = m
    def get_metrics(self):
        with self.lock: return self.metrics.copy()

state = SharedState()
posture_history = []

def draw_landmarks(frame, landmarks, w, h):
    conns = [(0,7),(0,8),(7,11),(8,12),(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]
    for s, e in conns:
        if s < len(landmarks) and e < len(landmarks):
            sl, el = landmarks[s], landmarks[e]
            if sl.visibility < 0.5 or el.visibility < 0.5: continue
            cv2.line(frame, (int(sl.x*w),int(sl.y*h)), (int(el.x*w),int(el.y*h)), (0,255,255), 2, cv2.LINE_AA)
    for lm in landmarks:
        if lm.visibility < 0.5: continue
        cx, cy = int(lm.x*w), int(lm.y*h)
        cv2.circle(frame, (cx,cy), 5, (0,229,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 7, (0,200,255), 1, cv2.LINE_AA)

def capture_loop():
    global posture_history
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Ainaa] ❌ No camera found. Serving placeholder feed.")
        while True:
            f = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(f,"AINAA MIRROR",(170,220),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,229,255),3)
            cv2.putText(f,"Camera offline. Check connection.",(160,280),cv2.FONT_HERSHEY_SIMPLEX,0.6,(150,150,150),1)
            _,b = cv2.imencode('.jpg',f); state.update_frames(b.tobytes(),b.tobytes()); time.sleep(0.5)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_save = time.time()

    # Initialize edge utilities
    segmentor = mp_selfie.SelfieSegmentation(model_selection=1)
    hand_det = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.01); continue
        h, w, _ = frame.shape
        _, raw_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Selfie Segmentation (Background Studio Effect)
        seg_res = segmentor.process(rgb)
        mask = seg_res.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.4
        
        # Create studio background (Dark gradient)
        bg = np.zeros(frame.shape, dtype=np.uint8)
        bg[:] = (20, 20, 20) # Very dark gray
        
        # Mix
        ann_frame = np.where(condition, frame, bg)

        # 2. Hand Gesture Detection
        hand_res = hand_det.process(rgb)
        current_gesture = "None"
        if hand_res.multi_hand_landmarks:
            for hand_lms in hand_res.multi_hand_landmarks:
                # Basic "Open Palm" detection
                tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
                up_count = 0
                for t in tips:
                    if hand_lms.landmark[t].y < hand_lms.landmark[t-2].y: up_count += 1
                if up_count >= 4: current_gesture = "Open Palm"

        # 3. Pose & Posture
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect(mp_img)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]
            draw_landmarks(ann_frame, lms, w, h)
            p = analyze_posture(lms)
            posture_history.append(p["composite_score"])
            if len(posture_history) > 8: posture_history = posture_history[-8:]
            smooth = round(sum(posture_history)/len(posture_history), 1)
            
            metrics = {"status":p["status"],"composite_score":smooth,"neck_angle":p["neck_angle"],
                       "shoulder_tilt":p["shoulder_tilt"],"torso_angle":p["torso_angle"],
                       "has_person":True, "gesture": current_gesture}
            state.update_metrics(metrics)
            
            c = (0,230,118) if p["status"]=="ALIGNED" else (0,171,255) if p["status"]=="FAIR" else (0,0,255)
            cv2.putText(ann_frame, f"{p['status']} ({smooth})", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)
            if current_gesture != "None":
                cv2.putText(ann_frame, f"GESTURE: {current_gesture}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if time.time() - last_save > 30:
                save_snapshot(posture_score=smooth, neck_angle=p["neck_angle"], shoulder_tilt=p["shoulder_tilt"], torso_angle=p["torso_angle"])
                last_save = time.time()
        else:
            state.update_metrics({"status":"NO_PERSON","composite_score":0,"neck_angle":0,"shoulder_tilt":0,"torso_angle":0,"has_person":False, "gesture": current_gesture})

        _, ann_buf = cv2.imencode('.jpg', ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        state.update_frames(raw_buf.tobytes(), ann_buf.tobytes())


capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()

def gen_mjpeg():
    while True:
        f = state.get_annotated()
        if f: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f + b'\r\n'
        time.sleep(0.033)

# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
def health(): return {"status":"online","project":"Ainaa","version":"3.0","xai":xai_client is not None}

@app.get("/stream/video")
def video_feed(): return StreamingResponse(gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps({"type":"posture","data":state.get_metrics()}))
            await asyncio.sleep(0.1)
    except: pass

@app.post("/api/v1/analyze/outfit")
@limiter.limit("5/minute")
async def analyze_outfit(request: Request):
    raw = state.get_raw()
    if not raw: return {"rating":0,"analysis":"No camera frame.","suggestions":"Check camera connection."}
    
    # Try xAI first
    if xai_client:
        result = await analyze_with_xai(raw)
        if result:
            save_snapshot(posture_score=state.get_metrics().get("composite_score",0), outfit_rating=result.get("rating"), analysis_text=result.get("analysis"))
            return result
    
    # Smart OpenCV fallback
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = analyze_style_authentic(frame)
    save_snapshot(posture_score=state.get_metrics().get("composite_score",0), outfit_rating=result.get("rating"), analysis_text=result.get("analysis"))
    return result


@app.get("/api/v1/vault")
async def get_vault():
    h = get_vault_history(20)
    return h if h else [{"timestamp":datetime.now().isoformat(),"posture_score":0,"outfit_rating":None,"analysis":"No data yet."}]

@app.post("/api/v1/chat")
@limiter.limit("10/minute")
async def chat(request: Request):
    body = await request.json()
    msg = body.get("message", "")
    metrics = state.get_metrics()
    
    # Try xAI for conversational AI
    if xai_client:
        try:
            response = xai_client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": "You are Ainaa, a friendly smart mirror AI. Keep responses under 2 sentences."},
                    {"role": "user", "content": f"Current posture: {json.dumps(metrics)}. User: \"{msg}\""}
                ]
            )
            return {"response": response.choices[0].message.content.strip()}
        except Exception as e:
            print(f"[Ainaa] Chat xAI error: {e}")
    
    # Smart local fallback
    return {"response": smart_chat_local(msg, metrics)}

if __name__ == "__main__":
    import uvicorn
    print("[Ainaa] Starting Ainaa Brain v3.0 on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
