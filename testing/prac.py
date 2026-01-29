import os
# Force Qt to use X11 (xcb) instead of Wayland to prevent crashes on Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import threading
import time
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def get_camera():
    for index in [0, 1, 2, -1]:
        # On Linux, usually just index is enough. DirectShow (CAP_DSHOW) is Windows-only.
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Success: Camera found at index {index}")
            return cap
    return None

class SurveillanceSystem:
    def __init__(self):
        self.frame = None
        self.annotated_frame = None  # Store the frame with drawing
        self.stopped = False
        self.lock = threading.Lock()
        
        self.cap = get_camera()
        if self.cap is None:
            raise IOError("Could not open video stream.")
        
    def start_camera(self):
        def update():
            while not self.stopped:
                success, img = self.cap.read()
                if success:
                    with self.lock:
                        self.frame = img
                else:
                    self.stopped = True
        threading.Thread(target=update, daemon=True).start()

    def run_face_ai(self):
        # RULE: Load model INSIDE the thread
        path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        model = YOLO(path)
        
        while not self.stopped:
            if self.frame is not None:
                with self.lock:
                    img_copy = self.frame.copy()
                # Run inference
                results = model.track(img_copy, persist=True, verbose=False)
                
                # Draw the results (bounding boxes) on the frame
                plotted_img = results[0].plot()

                with self.lock:
                    self.annotated_frame = plotted_img
            time.sleep(0.01) # Small sleep to prevent 100% CPU usage

# 1. Start System
sys = SurveillanceSystem()

sys.start_camera()

# 2. WAIT for the first frame (Fixes NoneType error)
print("Waiting for camera initialization...")
while sys.frame is None:
    time.sleep(0.1)

# 3. Start AI Threads
threading.Thread(target=sys.run_face_ai, daemon=True).start()

# 4. Main Display Loop
while not sys.stopped:
    display_frame = None
    with sys.lock:
        # Prefer the annotated frame if available, otherwise raw frame
        if sys.annotated_frame is not None:
            display_frame = sys.annotated_frame
        elif sys.frame is not None:
            display_frame = sys.frame

    if display_frame is not None:
        cv2.imshow("Surveillance", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.stopped = True
        break

sys.cap.release()
cv2.destroyAllWindows()