import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# --- 1. Load the Scout Model (YOLO) ---
# This model is trained just to find the bounding box of a "Weapon"
scout_model = YOLO('best_scout_weapon_detector.pt') 

# --- 2. Build & Load the Specialist Model (MobileNetV3) ---
# We use MobileNetV3-Small because it is incredibly fast on edge hardware
specialist_model = models.mobilenet_v3_small(weights=None)

# Modify the final classification head to output our 3 specific classes
num_features = specialist_model.classifier[3].in_features
specialist_model.classifier[3] = nn.Linear(num_features, 3)

# Load your custom trained weights for the specialist
# specialist_model.load_state_dict(torch.load('specialist_weights.pth'))
specialist_model.eval() # Set to evaluation mode

# Define the exact classes
CLASSES = ['Pistol', 'Rifle', 'Knife']

# --- 3. Define the Mathematical Tensor Transforms ---
# The specialist model expects a very specific mathematical format:
# A 224x224 RGB tensor, normalized to ImageNet standards.
crop_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_cascade(video_path=0):
    """Runs the cascading inference loop on a video feed."""
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # STEP A: The Scout Layer (Detect generic weapon)
        results = scout_model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Safety check: Ensure crop stays within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # STEP B: The Crop
                # Extract the weapon geometry mathematically using numpy slicing
                weapon_crop = frame[y1:y2, x1:x2]
                
                # Skip if the crop is impossibly small (prevents tensor errors)
                if weapon_crop.shape[0] < 10 or weapon_crop.shape[1] < 10:
                    continue
                    
                # STEP C: Tensor Transformation
                # Convert the BGR OpenCV image to RGB, then to a PyTorch Tensor
                rgb_crop = cv2.cvtColor(weapon_crop, cv2.COLOR_BGR2RGB)
                input_tensor = crop_transform(rgb_crop).unsqueeze(0) # Add batch dimension
                
                # STEP D: The Specialist Layer (Classify the specific weapon)
                with torch.no_grad(): # Disable gradient tracking for speed
                    outputs = specialist_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, class_idx = torch.max(probabilities, dim=0)
                
                final_class = CLASSES[class_idx.item()]
                final_conf = confidence.item()
                
                # --- Visual Output ---
                # Draw the bounding box and the Specialist's specific classification
                label = f"{final_class} {final_conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Cascaded Weapon Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Pass '0' for live webcam, or a path to a video file
    run_cascade(0)