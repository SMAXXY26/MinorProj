import os
# Force Qt to use X11 (xcb) instead of Wayland to prevent crashes on Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import threading
import time
from ultralytics import YOLO
import mediapipe as mp
from huggingface_hub import hf_hub_download

# --- 1. THE AI WORKER CLASS ---
class VisionWorker:
    def __init__(self, mode="face"):
        self.mode = mode
        self.frame = None
        self.results = None
        self.lock = threading.Lock()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        # Initialize models INSIDE the thread
        if self.mode == "face":
            path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
            model = YOLO(path)
        else:
            mp_pose = mp.solutions.pose
            model = mp_pose.Pose(model_complexity=0) # Lite for speed

        while not self.stopped:
            if self.frame is not None:
                with self.lock:
                    local_frame = self.frame.copy()
                
                if self.mode == "face":
                    # YOLO Inference
                    self.results = model.track(local_frame, persist=True, verbose=False, conf=0.5)
                else:
                    # MediaPipe Inference
                    img_rgb = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                    self.results = model.process(img_rgb)
            
            time.sleep(0.01)

# --- 2. INITIALIZE HARDWARE & THREADS ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

face_ai = VisionWorker(mode="face").start()
pose_ai = VisionWorker(mode="pose").start()
mp_drawing = mp.solutions.drawing_utils

print("Surveillance System Initializing...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Update both threads with the current frame
    face_ai.frame = frame
    pose_ai.frame = frame

    # --- DRAW POSE (SKELETON) ---
    with pose_ai.lock:
        if pose_ai.results and pose_ai.results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_ai.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # --- DRAW FACE (YOLO) ---
    with face_ai.lock:
        if face_ai.results and len(face_ai.results) > 0:
            face_results = face_ai.results[0].boxes
            if face_results is not None and face_results.id is not None:
                for box, id in zip(face_results.xyxy, face_results.id):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"FACE_{int(id)}", (x1, y1-10), 1, 1, (0, 255, 0), 2)

    # UI Feed
    cv2.putText(frame, "DUAL-AI SURVEILLANCE ACTIVE", (10, 30), 1, 1, (255, 255, 255), 2)
    cv2.imshow("CrowdWatch AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        face_ai.stopped = True
        pose_ai.stopped = True
        break

cap.release()
cv2.destroyAllWindows()