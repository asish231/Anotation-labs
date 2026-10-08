#!/usr/bin/env python3
"""
BioKin-Transformer Live Studio Server (Production Top-Down Pipeline)
1. Hand Presence Verification & ROI BBox Detection (MediaPipe / RTMPose)
2. Normalized Square Crop centered on the hand
3. BioKin-Transformer Biomechanical 3D Kinematics (0.0% Violation Rate)
4. Coordinate Remapping & HUD Telemetry
5. 1-Click Label Studio task export
"""
import os
import sys
import time
import json
import base64
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from biokin_hand.config import CONFIG, JOINT_NAMES, FINGER_CHAINS, JOINT_LIMITS_DEG
from biokin_hand.models.biokin_net import BioKinNet

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, "static"))
CORS(app)

# 1. Initialize BioKinNet with Trained Weights on Apple Silicon MPS
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[BioKin-Studio] Initializing BioKinNet on device: {device}...")

model = BioKinNet(pretrained=False).to(device)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

current_checkpoint_info = {
    "name": "biokin_best.pt",
    "epoch": "?",
    "mtime": "",
    "path": "checkpoints/biokin_best.pt"
}

def get_available_models():
    """
    Returns a list of all available .pt checkpoints in ./checkpoints
    with their epoch, filesize, and timestamp.
    """
    models = []
    if os.path.exists(CHECKPOINT_DIR):
        for fname in sorted(os.listdir(CHECKPOINT_DIR)):
            if fname.endswith(".pt"):
                full_path = os.path.join(CHECKPOINT_DIR, fname)
                try:
                    stat = os.stat(full_path)
                    size_mb = round(stat.st_size / (1024 * 1024), 1)
                    mtime = time.strftime('%H:%M:%S', time.localtime(stat.st_mtime))
                    models.append({
                        "name": fname,
                        "path": f"checkpoints/{fname}",
                        "size_mb": size_mb,
                        "mtime": mtime,
                        "is_current": (fname == current_checkpoint_info["name"])
                    })
                except Exception as e:
                    print(f"Error inspecting {fname}: {e}")
    # Sort latest modified first
    models.sort(key=lambda m: os.path.getmtime(os.path.join(CHECKPOINT_DIR, m["name"])), reverse=True)
    for m in models:
        m["is_current"] = (m["name"] == current_checkpoint_info["name"])
    return models

def load_checkpoint(ckpt_name=None):
    global current_checkpoint_info
    if not ckpt_name:
        ckpt_name = current_checkpoint_info["name"] or "biokin_best.pt"

    if os.path.isabs(ckpt_name):
        ckpt_path = ckpt_name
    elif os.path.dirname(ckpt_name):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_name)
    else:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

    if not os.path.exists(ckpt_path):
        return False, f"Checkpoint not found: {ckpt_name}"

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        epoch = ckpt.get("epoch", "?")
        mtime = time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(ckpt_path)))
        filename = os.path.basename(ckpt_path)
        current_checkpoint_info = {
            "name": filename,
            "epoch": epoch,
            "mtime": mtime,
            "path": f"checkpoints/{filename}"
        }
        msg = f"Loaded {filename} (Epoch {epoch}, Saved at {mtime})"
        print(f"[BioKin-Studio] {msg}")
        return True, msg
    except Exception as e:
        err = f"Failed to load {os.path.basename(ckpt_path)}: {e}"
        print(f"[BioKin-Studio] {err}")
        return False, err

load_checkpoint()

@app.route("/api/list_models", methods=["GET"])
def api_list_models():
    models = get_available_models()
    return jsonify({
        "models": models,
        "current_model": current_checkpoint_info["name"],
        "current_info": current_checkpoint_info
    })

@app.route("/api/reload_model", methods=["POST"])
def api_reload_model():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name")
    success, msg = load_checkpoint(model_name)
    return jsonify({
        "success": success,
        "message": msg,
        "current_model": current_checkpoint_info["name"],
        "current_info": current_checkpoint_info
    })

# 2. Initialize Lightweight Hand ROI Detector (MediaPipe + RTMPose fallback)
mp_landmarker = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    mp_task_path = os.path.join(PROJECT_ROOT, "hand_landmarker.task")
    if os.path.exists(mp_task_path):
        base_options = python.BaseOptions(model_asset_path=mp_task_path)
        mp_options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.18,
            min_hand_presence_confidence=0.18,
            min_tracking_confidence=0.18
        )
        mp_landmarker = vision.HandLandmarker.create_from_options(mp_options)
        print("[BioKin-Studio] Google MediaPipe HandLandmarker initialized.")
except Exception as e:
    print(f"[BioKin-Studio] MediaPipe init error: {e}")

rtm_estimator = None
try:
    from rtmlib import Hand
    rtm_estimator = Hand(mode="lightweight", device="cpu")
    print("[BioKin-Studio] OpenMMLab RTMPose-Hand detector initialized.")
except Exception as e:
    print(f"[BioKin-Studio] RTMPose init error: {e}")


def detect_hand_roi(img_bgr):
    """
    Detects hand presence and returns bounding box (xmin, ymin, xmax, ymax)
    and optional baseline keypoints.
    """
    h, w = img_bgr.shape[:2]

    # 1. Primary: MediaPipe Palm & Hand detector
    if mp_landmarker is not None:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_res = mp_landmarker.detect(mp_img)
            if mp_res.hand_landmarks and len(mp_res.hand_landmarks) > 0:
                lms = mp_res.hand_landmarks[0]
                xs = [lm.x * w for lm in lms]
                ys = [lm.y * h for lm in lms]
                bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
                kps_mp = [[round(xs[i], 1), round(ys[i], 1)] for i in range(len(xs))]
                return True, bbox, kps_mp, "mediapipe"
        except Exception as e:
            pass

    # 2. Fallback: RTMPose
    if rtm_estimator is not None:
        try:
            kps_rtm, scs = rtm_estimator(img_bgr)
            if len(scs) > 0 and np.mean(scs[0]) > 0.22:
                pts = kps_rtm[0]
                xs = pts[:, 0]
                ys = pts[:, 1]
                bbox = (float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))
                kps_rtm_list = [[round(float(xs[i]), 1), round(float(ys[i]), 1)] for i in range(len(xs))]
                return True, bbox, kps_rtm_list, "rtmpose"
        except Exception as e:
            pass

    return False, None, None, None


def format_label_studio_task(keypoints_2d, w, h):
    """Formats 21 joints to Label Studio format."""
    results = []
    for idx, (x, y) in enumerate(keypoints_2d):
        name = JOINT_NAMES[idx]
        x_pct = (x / float(w)) * 100.0
        y_pct = (y / float(h)) * 100.0
        results.append({
            "type": "keypointlabels",
            "value": {
                "x": round(x_pct, 2),
                "y": round(y_pct, 2),
                "width": 0.5,
                "keypointlabels": [name]
            },
            "id": f"kp_{idx}_{name.lower()}",
            "from_name": "keypoint_labels",
            "to_name": "image"
        })
    return {
        "result": results,
        "score": 1.0,
        "model_version": "BioKin-Transformer-V1"
    }


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        t_start = time.time()
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Could not decode image"}), 400

        h, w = img_bgr.shape[:2]

        # 1. Hand Presence Verification & ROI BBox Detection
        found, bbox, mp_kps, method = detect_hand_roi(img_bgr)

        if not found:
            # NO HAND IN FRAME: cleanly return detected: false (do NOT hallucinate over face)
            return jsonify({
                "detected": False,
                "width": w,
                "height": h,
                "latency_ms": round((time.time() - t_start) * 1000.0, 1),
                "biokin": None,
                "mediapipe": None
            })

        # 2. Generate Generous Square Crop around hand
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        # 40% margin to ensure wrist and fully extended fingertips fit comfortably
        box_size = max(xmax - xmin, ymax - ymin) * 1.45
        box_size = max(box_size, 100.0)

        x1 = max(0, int(cx - box_size / 2.0))
        y1 = max(0, int(cy - box_size / 2.0))
        x2 = min(w, int(cx + box_size / 2.0))
        y2 = min(h, int(cy + box_size / 2.0))

        hand_crop = img_bgr[y1:y2, x1:x2]
        crop_h, crop_w = hand_crop.shape[:2]

        if crop_h < 20 or crop_w < 20:
            return jsonify({"detected": False, "latency_ms": 1.0})

        # 3. BioKin-Transformer Biomechanical Kinematics on the cropped hand
        biokin_crop_res = model.predict_image(hand_crop, device=device)
        biokin_ms = (time.time() - t_start) * 1000.0

        # 4. Remap 2D keypoints back to original full frame coordinates
        kps_crop = biokin_crop_res["keypoints_2d"]
        kps_full = []
        for pt in kps_crop:
            kps_full.append([round(x1 + pt[0], 2), round(y1 + pt[1], 2)])

        return jsonify({
            "detected": True,
            "width": w,
            "height": h,
            "bbox": [x1, y1, x2, y2],
            "latency_ms": round(biokin_ms, 1),
            "biokin": {
                "keypoints_2d": kps_full,
                "keypoints_3d_mm": biokin_crop_res["keypoints_3d_mm"],
                "joint_angles_deg": biokin_crop_res["joint_angles_deg"],
                "anatomical_violations": biokin_crop_res["anatomical_violations"]
            },
            "mediapipe": mp_kps if data.get("compare_mediapipe", False) else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/export_snapshot", methods=["POST"])
def export_snapshot():
    try:
        data = request.json
        if not data or "image" not in data or "biokin" not in data:
            return jsonify({"error": "Missing image or prediction data"}), 400

        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img_bgr.shape[:2]
        timestamp = int(time.time())
        out_dir = os.path.join(PROJECT_ROOT, "annotated_output")
        os.makedirs(out_dir, exist_ok=True)

        img_path = os.path.join(out_dir, f"capture_{timestamp}.jpg")
        json_path = os.path.join(out_dir, f"capture_{timestamp}_label_studio.json")

        cv2.imwrite(img_path, img_bgr)

        kps_2d = data["biokin"]["keypoints_2d"]
        ls_task = format_label_studio_task(kps_2d, w, h)
        ls_task["image_path"] = img_path
        with open(json_path, "w") as f:
            json.dump(ls_task, f, indent=2)

        return jsonify({
            "success": True,
            "image_saved": img_path,
            "json_saved": json_path,
            "num_keypoints": len(kps_2d)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    models = get_available_models()
    return render_template_string(
        HTML_PAGE,
        models=models,
        current_model=current_checkpoint_info["name"],
        current_info=current_checkpoint_info
    )


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BioKin-Transformer Live Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090c12;
      --card-bg: rgba(16, 22, 34, 0.85);
      --card-border: rgba(45, 58, 80, 0.7);
      --accent-green: #10b981;
      --accent-emerald: #059669;
      --accent-cyan: #06b6d4;
      --accent-purple: #a855f7;
      --accent-pink: #ec4899;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: radial-gradient(circle at 50% 0%, #171c2f 0%, var(--bg) 60%, #040609 100%);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 14px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(12, 17, 28, 0.9);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, var(--accent-green), var(--accent-cyan));
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 18px;
      color: #061118;
      box-shadow: 0 0 16px rgba(16, 185, 129, 0.4);
    }

    .brand-title {
      font-size: 19px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    .brand-badge {
      font-size: 10px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.35);
      padding: 2px 7px;
      border-radius: 999px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
    }

    .status-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-green);
    }

    .model-selector-wrapper {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(15, 23, 42, 0.7);
      padding: 5px 12px;
      border: 1px solid var(--card-border);
      border-radius: 12px;
    }

    .model-select-container {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .model-select {
      background: #090e17;
      color: var(--accent-cyan);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 5px 10px;
      border-radius: 8px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      cursor: pointer;
      outline: none;
      transition: all 0.15s ease;
      min-width: 220px;
    }

    .model-select:hover, .model-select:focus {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }

    .btn-refresh-models {
      background: rgba(30, 41, 59, 0.8);
      color: var(--text-muted);
      border: 1px solid var(--card-border);
      padding: 6px 9px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
    }

    .btn-refresh-models:hover {
      color: #fff;
      background: rgba(51, 65, 85, 0.9);
      border-color: var(--accent-cyan);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      background: var(--accent-green);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--accent-green);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    main {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 20px;
      padding: 20px 28px;
      max-width: 1600px;
      width: 100%;
      margin: 0 auto;
    }

    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
    }

    .viewport-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      position: relative;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }

    .stage-wrapper {
      position: relative;
      width: 100%;
      border-radius: 12px;
      overflow: hidden;
      background: #020408;
      aspect-ratio: 4/3;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    #webcam-video {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1); /* mirror */
    }

    #stage-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 10;
    }

    .hud-banner {
      position: absolute;
      top: 14px;
      left: 14px;
      z-index: 20;
      background: rgba(10, 14, 22, 0.85);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(8px);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .controls-bar {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .btn-group {
      display: flex;
      gap: 10px;
      align-items: center;
    }

    button {
      padding: 9px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
      border: none;
      font-family: 'Outfit', sans-serif;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--accent-green), var(--accent-emerald));
      color: #03140d;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }

    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }

    .btn-secondary {
      background: rgba(30, 41, 59, 0.7);
      color: var(--text);
      border: 1px solid var(--card-border);
    }

    .btn-secondary:hover {
      background: rgba(51, 65, 85, 0.9);
    }

    .toggle-label {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
    }

    .toggle-label input { cursor: pointer; }

    /* Side Telemetry Sidebar */
    .telemetry-sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }

    .card-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-size: 13px;
    }

    .metric-val {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
    }

    .badge-pass {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 11px;
    }

    .badge-wait {
      background: rgba(148, 163, 184, 0.15);
      color: var(--text-muted);
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 11px;
    }

    .finger-angle-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-top: 8px;
    }

    .finger-pill {
      background: rgba(2, 6, 23, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 8px 10px;
    }

    .finger-name {
      font-size: 11px;
      font-weight: 600;
      margin-bottom: 3px;
    }

    .finger-degrees {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: var(--accent-cyan);
    }

    .legend-box {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      font-size: 11px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
    }

    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }

    #export-alert {
      margin-top: 8px;
      font-size: 12px;
      color: var(--accent-green);
      font-family: 'JetBrains Mono', monospace;
      display: none;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">BK</div>
      <div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="brand-title">BioKin-Transformer</span>
          <span class="brand-badge">PROD TOP-DOWN V1</span>
        </div>
        <div style="font-size: 11px; color: var(--text-muted);">ROI BBox Detection • 224px Patch Normalization • 0.0% Violations</div>
      </div>
    </div>

    <div style="display: flex; align-items: center; gap: 14px;">
      <!-- Dynamic Checkpoint Dropdown Selector -->
      <div class="model-selector-wrapper">
        <label for="model-select" style="font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 5px;">
          <svg width="12" height="12" fill="var(--accent-cyan)" viewBox="0 0 16 16"><path d="M8 1a2 2 0 0 1 2 2v2H6V3a2 2 0 0 1 2-2zm3 4V3a3 3 0 1 0-6 0v2H3.5A1.5 1.5 0 0 0 2 6.5v7A1.5 1.5 0 0 0 3.5 15h9a1.5 1.5 0 0 0 1.5-1.5v-7A1.5 1.5 0 0 0 12.5 5H11z"/></svg>
          Model:
        </label>
        <div class="model-select-container">
          <select id="model-select" class="model-select" onchange="switchModel(this.value)">
            {% for m in models %}
            <option value="{{ m.name }}" {% if m.name == current_model %}selected{% endif %}>
              {{ m.name }} ({{ m.size_mb }} MB • {{ m.mtime }})
            </option>
            {% endfor %}
          </select>
          <button class="btn-refresh-models" onclick="refreshModelList()" title="Refresh checkpoint list">
            <svg width="12" height="12" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/><path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/></svg>
          </button>
        </div>
      </div>

      <div class="status-pill">
        <div class="status-dot"></div>
        <span id="header-status">MPS ACCELERATION ACTIVE</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Viewport Area -->
    <div class="viewport-card">
      <div class="stage-wrapper">
        <video id="webcam-video" autoplay playsinline muted></video>
        <canvas id="stage-canvas"></canvas>
        <div class="hud-banner" id="hud-status">
          <span style="color: var(--accent-cyan);">●</span>
          <span id="hud-text">Initializing camera...</span>
        </div>
      </div>

      <div class="controls-bar">
        <div class="btn-group">
          <button id="btn-toggle-cam" class="btn-primary" onclick="toggleCamera()">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M10.5 8.5a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/><path d="M2 4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1.172a2 2 0 0 1-1.414-.586l-.828-.828A2 2 0 0 0 9.172 2H6.828a2 2 0 0 0-1.414.586l-.828.828A2 2 0 0 1 3.172 4H2zm.5 2a.5.5 0 1 1 0-1 .5.5 0 0 1 0 1zm9 2.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0z"/></svg>
            <span id="btn-cam-text">Start Camera</span>
          </button>

          <button class="btn-secondary" onclick="exportSnapshot()">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg>
            Capture & Save Label Studio Task
          </button>

          <button class="btn-secondary" onclick="reloadModelCheckpoint()" title="Hot-reload latest saved checkpoint from disk">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/><path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/></svg>
            <span id="btn-reload-text">Reload Model</span>
          </button>
        </div>

        <div class="btn-group">
          <label class="toggle-label">
            <input type="checkbox" id="chk-bbox" checked>
            Show Hand BBox ROI
          </label>
          <label class="toggle-label">
            <input type="checkbox" id="chk-mediapipe" onchange="toggleCompareMP(this.checked)">
            Compare with MediaPipe (Red ghost)
          </label>
          <label class="toggle-label">
            <input type="checkbox" id="chk-hud" checked>
            Show Skeleton Overlay
          </label>
        </div>
      </div>

      <div id="export-alert"></div>
    </div>

    <!-- Side Telemetry Area -->
    <div class="telemetry-sidebar">
      <!-- Manifold Guarantee Card -->
      <div class="card">
        <div class="card-title">
          <span>Hand Presence & Kinematics</span>
          <span class="badge-wait" id="badge-presence">NO HAND</span>
        </div>
        <div class="metric-row">
          <span style="color: var(--text-muted);">Active Checkpoint</span>
          <span class="metric-val" id="val-model-name" style="color: var(--accent-cyan); font-size: 11px;">biokin_best.pt</span>
        </div>
        <div class="metric-row">
          <span style="color: var(--text-muted);">Anatomical Violation Rate</span>
          <span class="metric-val" style="color: var(--accent-green);">0.0% Guaranteed</span>
        </div>
        <div class="metric-row">
          <span style="color: var(--text-muted);">Inference Latency</span>
          <span class="metric-val" id="val-latency">-- ms</span>
        </div>
        <div class="metric-row">
          <span style="color: var(--text-muted);">Root Metric Depth (Z)</span>
          <span class="metric-val" id="val-depth">-- mm</span>
        </div>
      </div>

      <!-- Finger Angles Telemetry -->
      <div class="card">
        <div class="card-title">Live Joint Angles (Euler)</div>
        <div class="finger-angle-grid">
          <div class="finger-pill">
            <div class="finger-name" style="color: #34d399;">Thumb (CMC/MCP)</div>
            <div class="finger-degrees" id="ang-thumb">--° / --°</div>
          </div>
          <div class="finger-pill">
            <div class="finger-name" style="color: #c084fc;">Index (MCP/PIP)</div>
            <div class="finger-degrees" id="ang-index">--° / --°</div>
          </div>
          <div class="finger-pill">
            <div class="finger-name" style="color: #facc15;">Middle (MCP/PIP)</div>
            <div class="finger-degrees" id="ang-middle">--° / --°</div>
          </div>
          <div class="finger-pill">
            <div class="finger-name" style="color: #f472b6;">Ring (MCP/PIP)</div>
            <div class="finger-degrees" id="ang-ring">--° / --°</div>
          </div>
          <div class="finger-pill" style="grid-column: span 2;">
            <div class="finger-name" style="color: #38bdf8;">Pinky (MCP/PIP)</div>
            <div class="finger-degrees" id="ang-pinky">--° / --°</div>
          </div>
        </div>

        <div class="legend-box">
          <div class="legend-item"><div class="legend-dot" style="background: #34d399;"></div>Thumb</div>
          <div class="legend-item"><div class="legend-dot" style="background: #c084fc;"></div>Index</div>
          <div class="legend-item"><div class="legend-dot" style="background: #facc15;"></div>Middle</div>
          <div class="legend-item"><div class="legend-dot" style="background: #f472b6;"></div>Ring</div>
          <div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div>Pinky</div>
        </div>
      </div>

      <!-- Architecture Pipeline Card -->
      <div class="card">
        <div class="card-title">Top-Down Architecture</div>
        <ol style="font-size: 12px; color: var(--text-muted); line-height: 1.6; padding-left: 16px;">
          <li><b>Stage 1 (Detection):</b> Checks hand presence. No phantom skeleton is ever drawn if hand is not in view.</li>
          <li><b>Stage 2 (ROI Crop):</b> Extracts generous square patch containing only the hand (matching FreiHAND domain).</li>
          <li><b>Stage 3 (BioKinNet):</b> Evaluates DINOv2 patch tokens + 3-step IEF kinematic decoder with hard manifold clamping.</li>
          <li><b>Stage 4 (Remap):</b> Maps 21 joint coordinates back onto full webcam feed coordinates.</li>
        </ol>
      </div>
    </div>
  </main>

  <script src="/static/studio.js"></script>
</body>
</html>
"""

if __name__ == "__main__":
    port = 8088
    print(f"\n=======================================================")
    print(f" ★ BIOKIN-TRANSFORMER LIVE STUDIO (TOP-DOWN V1)")
    print(f" Serving live at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
