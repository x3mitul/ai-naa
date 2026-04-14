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

# Legacy solutions for easier pixel access
mp_selfie = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


# --- Gemini AI (new SDK) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = None

if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        # Quick connectivity check
        gemini_client.models.generate_content(model='gemini-2.0-flash', contents='ping')
        print("[Ainaa] ✅ Gemini AI connected and verified.")
    except Exception as e:
        print(f"[Ainaa] ⚠️  Gemini AI error: {e}")
        print("[Ainaa] ⚠️  Falling back to smart local analysis.")
        gemini_client = None
else:
    print("[Ainaa] ℹ️  No GEMINI_API_KEY. Using smart local analysis.")

# --- App ---
app = FastAPI(title="Ainaa Backend", version="3.0.0")
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

# ============================================================
# HAIRSTYLE & SUNGLASSES RECOMMENDATION DATA
# ============================================================

HAIR_SUGGESTIONS = {
    "Oval": {
        "recommended": [
            {"name": "Beach Waves", "description": "Effortless loose waves that elongate and add movement — perfect for your balanced proportions.", "tags": ["versatile", "romantic", "modern"]},
            {"name": "Layered Lob", "description": "A long bob with layers that frames any oval face beautifully and suits all textures.", "tags": ["classic", "chic", "low-maintenance"]},
            {"name": "Textured Pixie", "description": "A short, textured cut that highlights your even features. Oval faces can rock almost any length.", "tags": ["bold", "edgy", "confident"]},
            {"name": "Sleek Middle Part", "description": "A clean center part with straight or slightly wavy hair elevates your natural symmetry.", "tags": ["minimalist", "elegant", "timeless"]},
        ],
        "avoid": [
            {"name": "Extremely High Volume at Crown", "reason": "Can make your face appear too elongated"},
        ],
        "sunglasses": [
            {"name": "Aviator", "reason": "The classic teardrop shape complements your balanced proportions flawlessly.", "vibe": "Classic"},
            {"name": "Wayfarer", "reason": "The bold rectangular frame adds edge without overwhelming your even features.", "vibe": "Bold"},
            {"name": "Round", "reason": "Circular frames create a retro, artistic look that suits oval faces perfectly.", "vibe": "Retro"},
        ]
    },
    "Round": {
        "recommended": [
            {"name": "High Crown with Side Part", "description": "Volume at the top elongates the face while a deep side part adds angular contrast.", "tags": ["structured", "elongating", "modern"]},
            {"name": "Long Layers", "description": "Long hair with subtle layers draws the eye downward, slimming a rounder silhouette.", "tags": ["classic", "slimming", "versatile"]},
            {"name": "Asymmetric Bob", "description": "A longer front, shorter back asymmetric bob creates sharp lines that contrast soft curves.", "tags": ["edgy", "chic", "structured"]},
            {"name": "Sleek Straight", "description": "Pin-straight hair adds verticality and definition when worn long past the chin.", "tags": ["minimalist", "elongating", "sleek"]},
        ],
        "avoid": [
            {"name": "Chin-Level Bob", "reason": "Ends at the widest part of the face, emphasizing roundness"},
            {"name": "Very Short Crops", "reason": "Can accentuate the circular face shape"},
        ],
        "sunglasses": [
            {"name": "Square", "reason": "Angular square frames create sharp contrast against soft round features.", "vibe": "Bold"},
            {"name": "Rectangle", "reason": "Rectangular lenses add width and structure, elongating a rounder face.", "vibe": "Structured"},
            {"name": "Geometric", "reason": "Polygonal frames introduce edgy angles that flatter rounded faces.", "vibe": "Edgy"},
        ]
    },
    "Square": {
        "recommended": [
            {"name": "Soft Curls", "description": "Loose curls or waves soften the jaw's strong angles with flowing, organic lines.", "tags": ["romantic", "softening", "feminine"]},
            {"name": "Side-Swept Bangs", "description": "Diagonal bangs break the symmetry of a square face and draw the eye inward.", "tags": ["classic", "softening", "chic"]},
            {"name": "Long Wavy Layers", "description": "Length past the shoulders with waves offsets wide jaws by narrowing the lower face visually.", "tags": ["elegant", "flowing", "versatile"]},
            {"name": "Textured Fringe", "description": "A wispy, tousled fringe adds softness above the eyebrows and rounds out brow lines.", "tags": ["modern", "casual", "softening"]},
        ],
        "avoid": [
            {"name": "Blunt Bob at Jaw", "reason": "Emphasizes the wide jaw and square angles"},
            {"name": "Severe Straight Cuts", "reason": "Rigid horizontal lines amplify the box shape"},
        ],
        "sunglasses": [
            {"name": "Round", "reason": "Circular frames soften the strong jaw and angular brow with gentle curves.", "vibe": "Soft"},
            {"name": "Oval", "reason": "A rounded oval lens is the perfect antidote to square architecture.", "vibe": "Classic"},
            {"name": "Aviator", "reason": "Teardrop shape with a slight curve beautifully counterbalances strong jawlines.", "vibe": "Timeless"},
        ]
    },
    "Heart": {
        "recommended": [
            {"name": "Chin-Length Bob", "description": "Ends right at the jaw to add width at the bottom, balancing a wider forehead.", "tags": ["balancing", "chic", "structured"]},
            {"name": "Side Part Waves", "description": "A deep side part with waves breaks up the wide forehead and draws attention downward.", "tags": ["romantic", "balancing", "classic"]},
            {"name": "Low Bun", "description": "A relaxed low bun or chignon draws attention to the jaw area, balancing the width above.", "tags": ["elegant", "updo", "minimalist"]},
            {"name": "Chin-Length Curly Bob", "description": "Curls at chin height add the visual volume needed to balance a heart-shaped face.", "tags": ["playful", "balancing", "textured"]},
        ],
        "avoid": [
            {"name": "Very Short Sides", "reason": "Emphasizes width at the forehead and temples"},
            {"name": "High Volume at Crown", "reason": "Adds more width where you already have it"},
        ],
        "sunglasses": [
            {"name": "Aviator", "reason": "Narrow at the top and wider at the bottom — perfect balance for a heart face.", "vibe": "Retro"},
            {"name": "Cat-Eye", "reason": "A light cat-eye lift adds symmetry without widening the already broad forehead.", "vibe": "Airy"},
            {"name": "Round", "reason": "Round frames keep emphasis at the center of the face, drawing away from the forehead.", "vibe": "Soft"},
        ]
    },
    "Diamond": {
        "recommended": [
            {"name": "Full Fringe Bangs", "description": "A full fringe adds width to the forehead, balancing the narrow forehead-to-cheekbone contrast.", "tags": ["bold", "balancing", "retro"]},
            {"name": "Side Sweep", "description": "A dramatic side sweep adds forehead width while keeping the look asymmetric and modern.", "tags": ["modern", "balancing", "chic"]},
            {"name": "Curly Bob", "description": "Curls ending at cheekbone level add width at the face's narrower parts.", "tags": ["textured", "playful", "structured"]},
            {"name": "Layered Medium", "description": "Mid-length layers with volume at the top and bottom frame and fill narrow ends.", "tags": ["versatile", "elegant", "balancing"]},
        ],
        "avoid": [
            {"name": "Sleek Center Part", "reason": "Draws attention to the narrow forehead and chin"},
            {"name": "Slicked Back", "reason": "Exposes all the narrow points of a diamond face without softening"},
        ],
        "sunglasses": [
            {"name": "Oval", "reason": "Gently curved oval lenses add softness and minimize angular cheekbones.", "vibe": "Elegant"},
            {"name": "Rimless", "reason": "Rimless frames are understated, letting your cheekbones shine without competing.", "vibe": "Minimal"},
            {"name": "Cat-Eye", "reason": "Upswept cat-eye frames broaden the forehead visually, balancing the diamond.", "vibe": "Sophisticated"},
        ]
    },
    "Oblong": {
        "recommended": [
            {"name": "Curtain Bangs", "description": "Wispy curtain bangs shorten the appearance of a long face by adding horizontal emphasis.", "tags": ["trendy", "shortening", "soft"]},
            {"name": "Voluminous Waves", "description": "Wide, voluminous waves add horizontal width to a narrow elongated face shape.", "tags": ["glamorous", "widening", "romantic"]},
            {"name": "Short Layered Bob", "description": "A chin-length layered bob with volume on the sides widens and shortens the face.", "tags": ["structured", "widening", "chic"]},
            {"name": "Side Part with Volume", "description": "A voluminous side part adds width and breaks the vertical line of a long face.", "tags": ["classic", "widening", "versatile"]},
        ],
        "avoid": [
            {"name": "Very Long Straight Hair", "reason": "Adds length and elongates the face further"},
            {"name": "Center Part Slicked Down", "reason": "Emphasizes the vertical length of the face"},
        ],
        "sunglasses": [
            {"name": "Oversized", "reason": "Large frames add horizontal emphasis, shortening the appearance of a long face.", "vibe": "Statement"},
            {"name": "Wraparound", "reason": "Wide wraparound styles add visual width across the face.", "vibe": "Sporty"},
            {"name": "Square", "reason": "Angular square frames interrupt the vertical flow, creating pleasing proportion.", "vibe": "Bold"},
        ]
    },
}


def classify_face_shape_facemesh(landmarks, w, h):
    """Use 468 FaceMesh landmarks to classify face shape with measurements."""
    # Key landmark indices
    # Forehead: 10 (top center)
    # Chin: 152 (bottom center)
    # Left cheek: 234, Right cheek: 454
    # Left jaw: 172, Right jaw: 397
    # Left brow outer: 70, Right brow outer: 300
    def pt(idx):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    top = pt(10)
    bottom = pt(152)
    left_cheek = pt(234)
    right_cheek = pt(454)
    left_jaw = pt(172)
    right_jaw = pt(397)
    left_brow = pt(70)
    right_brow = pt(300)

    face_height = abs(bottom[1] - top[1])
    face_width = abs(right_cheek[0] - left_cheek[0])  # widest cheekbone width
    jaw_width = abs(right_jaw[0] - left_jaw[0])
    forehead_width = abs(right_brow[0] - left_brow[0])

    if face_height < 1 or face_width < 1:
        return "Oval", 0.6, {}

    aspect_ratio = face_height / face_width
    jaw_forehead_ratio = jaw_width / (forehead_width + 1e-6)
    cheek_jaw_ratio = face_width / (jaw_width + 1e-6)
    cheek_forehead_ratio = face_width / (forehead_width + 1e-6)

    measurements = {
        "aspect_ratio": round(aspect_ratio, 2),
        "jaw_forehead_ratio": round(jaw_forehead_ratio, 2),
        "cheek_jaw_ratio": round(cheek_jaw_ratio, 2),
        "cheek_forehead_ratio": round(cheek_forehead_ratio, 2),
    }

    # Classification logic (geometric heuristics)
    if aspect_ratio > 1.55:
        shape, conf = "Oblong", 0.82
    elif aspect_ratio < 1.15:
        shape, conf = "Round", 0.80
    elif cheek_jaw_ratio > 1.25 and cheek_forehead_ratio > 1.15:
        shape, conf = "Diamond", 0.77
    elif jaw_forehead_ratio < 0.80:
        shape, conf = "Heart", 0.78
    elif jaw_forehead_ratio > 0.95 and aspect_ratio < 1.40:
        shape, conf = "Square", 0.80
    else:
        shape, conf = "Oval", 0.85

    return shape, conf, measurements


def build_hairstyle_response(face_shape, confidence, measurements):
    """Build the complete hairstyle + sunglasses response."""
    data = HAIR_SUGGESTIONS.get(face_shape, HAIR_SUGGESTIONS["Oval"])
    return {
        "face_shape": face_shape,
        "confidence": confidence,
        "measurements": measurements,
        "hairstyle_recommendations": data["recommended"],
        "hairstyles_to_avoid": data["avoid"],
        "sunglasses_recommendations": data["sunglasses"],
    }


# ============================================================
# SMART OUTFIT ANALYSIS (OpenCV-based when no API key)
# ============================================================
def analyze_outfit_cv(frame):
    """Rich OpenCV-based outfit analysis analyzing colors, contrast, and composition."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define body regions
    face_region = hsv[int(h*0.05):int(h*0.25), int(w*0.3):int(w*0.7)]
    upper_body = hsv[int(h*0.25):int(h*0.55), int(w*0.2):int(w*0.8)]
    lower_body = hsv[int(h*0.55):int(h*0.85), int(w*0.2):int(w*0.8)]

    def get_color_info(region):
        if region.size == 0:
            return "neutral", "medium", 0
        hue_names = {0:"Red", 1:"Orange", 2:"Yellow", 3:"Lime", 4:"Green", 5:"Teal",
                     6:"Cyan", 7:"Azure", 8:"Blue", 9:"Purple", 10:"Magenta", 11:"Rose"}
        hist = cv2.calcHist([region], [0], None, [12], [0, 180])
        dom_hue = int(np.argmax(hist))
        avg_sat = float(np.mean(region[:, :, 1]))
        avg_val = float(np.mean(region[:, :, 2]))
        color = hue_names.get(dom_hue, "Neutral")
        brightness = "bright" if avg_val > 170 else "medium" if avg_val > 85 else "dark"
        return color, brightness, avg_sat

    upper_color, upper_bright, upper_sat = get_color_info(upper_body)
    lower_color, lower_bright, lower_sat = get_color_info(lower_body)

    # Scoring logic
    score = 6.5
    details = []

    # Color harmony
    if upper_color != lower_color:
        score += 1.0
        details.append(f"Nice contrast between your {upper_bright} {upper_color} top and {lower_bright} {lower_color} bottom")
    else:
        details.append(f"You're going monochromatic with {upper_color} tones — clean and intentional")
        score += 0.5

    # Saturation variety
    sat_diff = abs(upper_sat - lower_sat)
    if sat_diff > 30:
        score += 0.5
        details.append("Good saturation balance creates visual depth")
    
    # Brightness contrast
    if upper_bright != lower_bright:
        score += 0.5
        details.append("The light-dark balance between pieces is well-executed")

    # Overall vibrancy
    avg_sat_total = (upper_sat + lower_sat) / 2
    if avg_sat_total > 80:
        score += 0.5
        details.append("The colors are vibrant and eye-catching")
    elif avg_sat_total < 30:
        details.append("The muted palette gives a sophisticated, understated look")
        score += 0.3

    score = round(max(4, min(10, score)), 1)
    analysis = details[0] + "." if details else "Looking put-together."
    
    suggestions = []
    if score < 7:
        suggestions.append("Try adding a pop of color with an accessory")
    if upper_bright == lower_bright == "dark":
        suggestions.append("A lighter accent piece could brighten the look")
    if not suggestions:
        suggestions.append("This outfit works well — own it with confidence")

    # MOCK DATA GENERATOR FOR HACKATHON DEMO (When Gemini API fails)
    face_shapes = ["Oval", "Square", "Round", "Diamond", "Heart"]
    skin_tones = ["Warm Olive", "Cool Fair", "Deep Warm", "Neutral Beige", "Rich Brown"]
    hairstyles = ["Textured Crop", "Messy Fringe", "Slicked Back", "Buzz Cut", "Flowing Locks"]
    eye_colors = ["Dark Brown", "Hazel", "Deep Blue", "Amber", "Green"]
    spectacles_map = {
        "Oval": "Geometric or rectangular frames",
        "Square": "Round or oval wireframes",
        "Round": "Angular, rectangular, or cat-eye frames",
        "Diamond": "Rimless or horn-rimmed glasses",
        "Heart": "Bottom-heavy or wide rectangular frames"
    }
    
    gen_face = random.choice(face_shapes)
    gen_specs = spectacles_map[gen_face]

    return {
        "rating": score,
        "analysis": analysis,
        "suggestions": suggestions[0],
        "skin_tone": "Analyzing...",
        "face_shape": "Analyzing...",
        "hairstyle": "Analyzing...",
        "eye_color": "Analyzing...",
        "color_suggestions": ["Navy Blue", "Olive Green", "Charcoal"],
        "spectacles": "Analyzing..."
    }

# ============================================================
# AUTHENTIC EDGE AI ENGINES
# ============================================================

def analyze_style_authentic(frame):
    """Authentic local CV analysis for styling traits."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    style_data = {
        "skin_tone": "Neutral", "face_shape": "Oval", 
        "hairstyle": "Modern", "eye_color": "Brown",
        "spectacles": "Rectangular frames",
        "color_suggestions": ["Navy", "Charcoal", "Emerald"]
    }

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_det:
        results = face_det.process(rgb)
        if results.detections:
            # Get primary face
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
            xw, xh = int(bbox.width * w), int(bbox.height * h)
            
            # 1. Face Shape Heuristic (Aspect Ratio)
            ratio = xw / (xh + 1e-6)
            if ratio > 0.85: style_data["face_shape"] = "Round"
            elif ratio < 0.75: style_data["face_shape"] = "Oval"
            else: style_data["face_shape"] = "Square"
            
            # 2. Skin Tone Sampling (Forehead area)
            f_y = max(0, ymin + int(xh * 0.15))
            f_h = int(xh * 0.1)
            forehead = frame[f_y:f_y+f_h, xmin+int(xw*0.4):xmin+int(xw*0.6)]
            if forehead.size > 0:
                avg_bgr = np.mean(forehead, axis=(0, 1))
                if avg_bgr[2] > 200: style_data["skin_tone"] = "Fair"
                elif avg_bgr[2] > 140: style_data["skin_tone"] = "Medium / Olive"
                else: style_data["skin_tone"] = "Deep"

            # 3. Hair Color Sampling (Top area)
            h_y = max(0, ymin - int(xh * 0.1))
            h_h = int(xh * 0.15)
            hair_zone = frame[h_y:ymin, xmin+int(xw*0.3):xmin+int(xw*0.7)]
            if hair_zone.size > 0:
                avg_h = np.mean(cv2.cvtColor(hair_zone, cv2.COLOR_BGR2HSV)[:,:,2])
                style_data["hairstyle"] = "Dark / Natural" if avg_h < 80 else "Light / Styled"

            # 4. Spectacles Recommendation
            spec_map = {"Round": "Angular Geometric", "Oval": "Rectangular", "Square": "Round Wire", "Diamond":"Rimless"}
            style_data["spectacles"] = spec_map.get(style_data["face_shape"], "Modern Wayfarer")

    # Merge with outfit analysis
    outfit = analyze_outfit_cv(frame)
    style_data.update(outfit)
    return style_data


async def analyze_with_gemini(frame_bytes):
    if not gemini_client:
        return None
    try:
        from google.genai import types
        b64 = base64.b64encode(frame_bytes).decode('utf-8')
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                types.Content(parts=[
                    types.Part(text="""You are Ainaa, a stylish AI mirror. Analyze this person's outfit and facial features.
Respond as raw JSON only: {"rating": <1-10>, "analysis": "<2 sentences>", "suggestions": "<1 sentence>", "skin_tone": "<e.g. Warm Olive>", "face_shape": "<e.g. Oval>", "hairstyle": "<e.g. Messy Fringe>", "eye_color": "<e.g. Dark Brown>", "color_suggestions": ["<color1>", "<color2>"], "spectacles": "<e.g. Round Tortoiseshell frames>"}"""),
                    types.Part(inline_data=types.Blob(mime_type='image/jpeg', data=base64.b64decode(b64)))
                ])
            ]
        )
        text = response.text.strip()
        if text.startswith("```"): text = text.split("\n",1)[1].rsplit("```",1)[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"[Ainaa] Gemini vision error: {e}")
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
    if any(w in msg for w in ['hairstyle', 'haircut', 'hair', 'sunglasses', 'glasses', 'sunnies', 'face shape']):
        return "I can detect your face shape and suggest the best hairstyles and sunglasses for you! Click the '✂️ Hair & Sunnies' button or say 'suggest a hairstyle' and I'll analyse your features."
    
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
        print("[Ainaa] ❌ No camera. Serving placeholder.")
        while True:
            f = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(f,"AINAA",(200,220),cv2.FONT_HERSHEY_SIMPLEX,2,(0,229,255),3)
            cv2.putText(f,"Awaiting Camera...",(180,280),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,100),1)
            _,b = cv2.imencode('.jpg',f); state.update_frames(b.tobytes(),b.tobytes()); time.sleep(0.1)

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
def health(): return {"status":"online","project":"Ainaa","version":"3.0","gemini":gemini_client is not None}

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
async def analyze_outfit():
    raw = state.get_raw()
    if not raw: return {"rating":0,"analysis":"No camera frame.","suggestions":"Check camera connection."}
    
    # Try Gemini first
    if gemini_client:
        result = await analyze_with_gemini(raw)
        if result:
            save_snapshot(posture_score=state.get_metrics().get("composite_score",0), outfit_rating=result.get("rating"), analysis_text=result.get("analysis"))
            return result
    
    # Smart OpenCV fallback
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = analyze_style_authentic(frame)
    save_snapshot(posture_score=state.get_metrics().get("composite_score",0), outfit_rating=result.get("rating"), analysis_text=result.get("analysis"))
    return result


@app.post("/api/v1/analyze/hairstyle")
async def analyze_hairstyle():
    """Detect face shape using MediaPipe FaceMesh and return hairstyle + sunglasses recommendations."""
    raw = state.get_raw()
    if not raw:
        # Fallback: no camera frame
        shape, conf, meas = "Oval", 0.5, {}
        return build_hairstyle_response(shape, conf, meas)
    
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return build_hairstyle_response("Oval", 0.5, {})
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Try FaceMesh first (most accurate)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    ) as fm:
        results = fm.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            shape, conf, meas = classify_face_shape_facemesh(landmarks, w, h)
            return build_hairstyle_response(shape, conf, meas)
    
    # Fallback: Simple bounding-box heuristic via face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        det_results = fd.process(rgb)
        if det_results.detections:
            bbox = det_results.detections[0].location_data.relative_bounding_box
            fw = bbox.width * w
            fh = bbox.height * h
            ratio = fh / (fw + 1e-6)
            if ratio > 1.45:   shape, conf = "Oblong", 0.65
            elif ratio < 1.1:  shape, conf = "Round",  0.65
            else:              shape, conf = "Oval",   0.65
            meas = {"aspect_ratio": round(ratio, 2)}
            return build_hairstyle_response(shape, conf, meas)
    
    # No face detected
    return {"error": "No face detected. Please face the camera directly.",
            **build_hairstyle_response("Oval", 0.0, {})}


@app.get("/api/v1/vault")
async def get_vault():
    h = get_vault_history(20)
    return h if h else [{"timestamp":datetime.now().isoformat(),"posture_score":0,"outfit_rating":None,"analysis":"No data yet."}]

@app.post("/api/v1/chat")
async def chat(request: Request):
    body = await request.json()
    msg = body.get("message", "")
    metrics = state.get_metrics()
    
    # Try Gemini for conversational AI
    if gemini_client:
        try:
            from google import genai
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=f"You are Ainaa, a friendly smart mirror AI. Keep responses under 2 sentences. Current posture: {json.dumps(metrics)}. User: \"{msg}\""
            )
            return {"response": response.text.strip()}
        except Exception as e:
            print(f"[Ainaa] Chat Gemini error: {e}")
    
    # Smart local fallback
    return {"response": smart_chat_local(msg, metrics)}

if __name__ == "__main__":
    import uvicorn
    print("[Ainaa] Starting Ainaa Brain v3.0 on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
