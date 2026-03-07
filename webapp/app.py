"""
Weapon Detection Web App
========================
Flask server that streams webcam through YOLOv8 weapon detection model,
draws bounding boxes, and serves live detection stats via SSE.

Usage:
    python webapp/app.py
    python webapp/app.py --model path/to/best.pt
    python webapp/app.py --port 8080
"""

import argparse
import json
import time
import threading
from pathlib import Path

import cv2
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Globals ──────────────────────────────────────────────────────────────────
model = None
camera = None
lock = threading.Lock()

# Detection statistics (thread-safe via lock)
stats = {
    "Knife": 0,
    "Pistol": 0,
    "Rifle": 0,
    "total": 0,
    "fps": 0.0,
    "frame_detections": 0,  # detections in current frame
}

# Class name mapping (from model training)
CLASS_NAMES = {1: "Knife", 3: "Pistol", 4: "Rifle"}

# Bounding box colors per class (BGR for OpenCV)
CLASS_COLORS = {
    "Knife":  (0, 165, 255),   # orange
    "Pistol": (0, 0, 255),     # red
    "Rifle":  (255, 50, 50),   # blue
}

CONFIDENCE_THRESHOLD = 0.5
PERSISTENCE_THRESHOLD = 1.0  # seconds an object must stay in frame to count
IOU_MATCH_THRESHOLD = 0.3    # IoU threshold to match boxes across frames
STALE_TIMEOUT = 0.5          # seconds before a lost object is removed


# ── Object Persistence Tracker ───────────────────────────────────────────────

class TrackedObject:
    """Tracks a single detected object across frames."""
    _next_id = 0

    def __init__(self, cls_name, bbox):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1
        self.cls_name = cls_name
        self.bbox = bbox            # (x1, y1, x2, y2)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.counted = False        # True once it's been counted in stats

    def update(self, bbox):
        self.bbox = bbox
        self.last_seen = time.time()

    @property
    def duration(self):
        return self.last_seen - self.first_seen

    @property
    def is_persistent(self):
        return self.duration >= PERSISTENCE_THRESHOLD

    @property
    def is_stale(self):
        return (time.time() - self.last_seen) > STALE_TIMEOUT


def compute_iou(box_a, box_b):
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


# Active tracked objects (managed inside generate_frames)
tracked_objects = []


def init_model(model_path):
    """Load the YOLOv8 model."""
    global model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    # Get actual class names from the model
    print(f"Model classes: {model.names}")
    return model


def init_camera(source=0):
    """Open webcam."""
    global camera
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera connection.")
    # Set resolution for smoother streaming
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera opened successfully.")
    return camera


def generate_frames():
    """Generator that yields MJPEG frames with YOLO detections drawn."""
    global stats, tracked_objects
    prev_time = time.time()

    while True:
        with lock:
            success, frame = camera.read()

        if not success:
            continue

        # Run inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Collect current frame's detections as (cls_name, bbox) tuples
        current_detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    if cls_name in CLASS_COLORS:
                        bbox = tuple(map(int, box.xyxy[0]))
                        conf = float(box.conf[0])
                        current_detections.append((cls_name, bbox, conf))

        # ── Match current detections to tracked objects via IoU ───────
        matched_track_ids = set()
        matched_det_ids = set()

        for det_idx, (cls_name, bbox, conf) in enumerate(current_detections):
            best_iou = 0
            best_track_idx = -1
            for track_idx, obj in enumerate(tracked_objects):
                if track_idx in matched_track_ids:
                    continue
                if obj.cls_name != cls_name:
                    continue
                iou = compute_iou(bbox, obj.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx

            if best_iou >= IOU_MATCH_THRESHOLD and best_track_idx >= 0:
                # Update existing tracked object
                tracked_objects[best_track_idx].update(bbox)
                matched_track_ids.add(best_track_idx)
                matched_det_ids.add(det_idx)
            else:
                # New object — start tracking
                new_obj = TrackedObject(cls_name, bbox)
                tracked_objects.append(new_obj)
                matched_det_ids.add(det_idx)

        # Remove stale tracked objects (not seen recently)
        tracked_objects = [obj for obj in tracked_objects if not obj.is_stale]

        # ── Count persistent detections & draw boxes ─────────────────
        frame_counts = {"Knife": 0, "Pistol": 0, "Rifle": 0}
        frame_total = 0
        new_confirmed = {"Knife": 0, "Pistol": 0, "Rifle": 0}

        for det_idx, (cls_name, bbox, conf) in enumerate(current_detections):
            x1, y1, x2, y2 = bbox
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))

            # Find the tracked object for this detection
            is_confirmed = False
            for obj in tracked_objects:
                if obj.cls_name == cls_name and compute_iou(bbox, obj.bbox) >= IOU_MATCH_THRESHOLD:
                    if obj.is_persistent:
                        is_confirmed = True
                        if not obj.counted:
                            obj.counted = True
                            new_confirmed[cls_name] += 1
                    break

            # Count in frame regardless of persistence (for frame_detections display)
            frame_counts[cls_name] += 1
            frame_total += 1

            # Draw bounding box with persistence indicator
            thickness = 3 if is_confirmed else 2
            if is_confirmed:
                label = f"{cls_name} {conf:.2f}"
            else:
                label = f"{cls_name} {conf:.2f} ..."

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Update stats — only add newly confirmed (persistent) detections
        with lock:
            for cls_name, count in new_confirmed.items():
                stats[cls_name] += count
            stats["total"] += sum(new_confirmed.values())
            stats["frame_detections"] = frame_total
            stats["fps"] = round(fps, 1)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats_feed')
def stats_feed():
    """Server-Sent Events stream for detection statistics."""
    def event_stream():
        while True:
            with lock:
                data = json.dumps(stats)
            yield f"data: {data}\n\n"
            time.sleep(0.3)  # update every 300ms

    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset all detection counters."""
    global stats, tracked_objects
    with lock:
        stats = {
            "Knife": 0,
            "Pistol": 0,
            "Rifle": 0,
            "total": 0,
            "fps": 0.0,
            "frame_detections": 0,
        }
        tracked_objects = []
    return jsonify({"status": "ok"})


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weapon Detection Web App")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/runs/gun_model8/weights/best.pt",
        help="Path to YOLOv8 weights file"
    )
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--camera", type=int, default=0, help="Camera source index")
    args = parser.parse_args()

    # Resolve model path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Available models:")
        for pt in project_root.glob("runs/detect/runs/*/weights/best.pt"):
            print(f"  - {pt.relative_to(project_root)}")
        return

    init_model(str(model_path))
    init_camera(args.camera)

    print(f"\n{'='*50}")
    print(f"  Weapon Detection Web App")
    print(f"  Open http://localhost:{args.port} in your browser")
    print(f"{'='*50}\n")

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
