import os
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Direct MediaPipe imports for stability
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- 1. SETUP MODELS ---
print("Loading Models...")
# Load YOLO Face
face_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(face_path)

# Load MediaPipe Pose (Lite version for speed)
pose_model = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)

# --- 2. CAMERA SETUP ---
cap = cv2.VideoCapture(0) # Use cv2.CAP_DSHOW here if you are on Windows
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting Sequential Feed. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- 3. RUN FACE AI (YOLO) ---
    # We use .predict() instead of .track() for simpler sequential logic
    face_results = face_model.predict(frame, conf=0.5, verbose=False)
    
    # Draw Face Boxes
    if face_results and len(face_results) > 0:
        for box in face_results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- 4. RUN POSE AI (MEDIAPIPE) ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_model.process(img_rgb)
    
    # Draw Skeleton
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

    # --- 5. DISPLAY ---
    cv2.putText(frame, "SEQUENTIAL MODE - LOW FPS", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("CrowdWatch AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()