import cv2
from ultralytics import YOLO
import os
import sys
import time
import math
import datetime

# Try importing huggingface_hub with a clear error message if missing
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: 'huggingface_hub' is not installed. Please run: pip install huggingface_hub")
    sys.exit(1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # --- 1. Load Object Detection Model (YOLO) ---
    print("\n--- Loading Object Detection Model ---")
    possible_models = [
        os.path.join(project_root, "yolo11n.pt"),
        os.path.join(project_root, "yolov8n.pt"),
        "yolo11n.pt",
        "yolov8n.pt"
    ]
    
    model_path = "yolov8n.pt" # Default fallback
    for path in possible_models:
        if os.path.exists(path):
            model_path = path
            break
            
    print(f"Loading Model from: {model_path}")
    try:
        # Load model for both Person (class 0) and Bottle (class 39)
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Camera Setup ---
    print("\n--- Initializing Camera ---")
    cap = None
    # Try index 1 first (as per your preference)
    try:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Camera index 1 failed or not found. Trying index 0...")
            cap = cv2.VideoCapture(0)
    except Exception as e:
        print(f"Exception opening camera 1: {e}")
        cap = cv2.VideoCapture(0)

    if not cap or not cap.isOpened():
        print("Error: Could not open any webcam (indices 1 or 0).")
        return

    # Set camera to 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n=== System Ready ===")
    print("Mode: Track ONLY Persons holding Bottles")
    print("Press 'q' to quit.")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame (stream end or device error).")
                time.sleep(0.1)
                continue

            # Track Persons (0) and Bottles (39)
            results = model.track(frame, persist=True, classes=[0, 39], verbose=False)
            
            # Prepare to draw manually
            annotated_frame = frame.copy()
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                
                persons = []
                bottles = []
                
                # Separate detections
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person
                        persons.append(box)
                    elif cls == 39:  # Bottle
                        bottles.append(box)
                
                # Determine which persons have bottles
                # A simple heuristic: if a bottle center is inside the person bbox
                persons_with_bottles = []
                
                for p_box in persons:
                    p_x1, p_y1, p_x2, p_y2 = p_box.xyxy[0].cpu().numpy()
                    
                    has_bottle = False
                    for b_box in bottles:
                        b_x1, b_y1, b_x2, b_y2 = b_box.xyxy[0].cpu().numpy()
                        b_center_x = (b_x1 + b_x2) / 2
                        b_center_y = (b_y1 + b_y2) / 2
                        
                        # Check if bottle center is within person bounding box
                        if (p_x1 < b_center_x < p_x2) and (p_y1 < b_center_y < p_y2):
                            has_bottle = True
                            break # Found a bottle for this person
                    
                    if has_bottle:
                        persons_with_bottles.append(p_box)

                # Reset annotations to just the relevant detections
                # We can use the ultralytics plotting utils for specific boxes, or draw manually for control
                # Let's draw manually to be precise about what we show
                
                img_h, img_w = annotated_frame.shape[:2]
                
                # Draw ALL bottles (context is useful)
                for b_box in bottles:
                    try:
                        b_xyxy = b_box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(annotated_frame, (b_xyxy[0], b_xyxy[1]), (b_xyxy[2], b_xyxy[3]), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "Bottle", (b_xyxy[0], b_xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    except: pass
                    
                # Draw ONLY persons with bottles
                for p_box in persons_with_bottles:
                    try:
                        p_xyxy = p_box.xyxy[0].cpu().numpy().astype(int)
                        # Draw Person Box
                        cv2.rectangle(annotated_frame, (p_xyxy[0], p_xyxy[1]), (p_xyxy[2], p_xyxy[3]), (0, 255, 0), 2)
                        
                        # Label
                        label = "Person with Bottle"
                        if p_box.id is not None:
                            label += f" ID:{int(p_box.id[0])}"
                            
                        cv2.putText(annotated_frame, label, (p_xyxy[0], p_xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except: pass
                    
                count_text = f"Persons with Bottle: {len(persons_with_bottles)}"
                cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Save frame if person with bottle is detected
                if len(persons_with_bottles) > 0:
                    output_dir = os.path.join(script_dir, "detections")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(output_dir, f"detect_{timestamp}.jpg")
                    cv2.imwrite(filename, frame) # Save original clean frame

            cv2.imshow('Smart Bottle Tracker', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Runtime error in loop: {e}")
            # continue # Optionally continue or break
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
