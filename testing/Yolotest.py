import cv2
from ultralytics import YOLO

# 1. Load the YOLO26n-face model
# The model will download automatically if not present locally
model = YOLO('yolo26n-face.pt') 

# 2. Open Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Inference
    # YOLO26 is NMS-free, so we don't need complex post-processing parameters
    results = model.predict(frame, conf=0.5, stream=True)

    for r in results:
        # Plot standard detections (boxes/labels)
        annotated_frame = r.plot() 

        # 4. Access Keypoints (Estimation)
        if r.keypoints is not None:
            # xy[0] contains the [x, y] pairs for one detected face
            # Landmark order: 0:Left Eye, 1:Right Eye, 2:Nose, 3:Left Mouth, 4:Right Mouth
            keypoints = r.keypoints.xy[0].cpu().numpy()

            for i, kp in enumerate(keypoints):
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:  # Only draw visible points
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, str(i), (x+5, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 5. Display
    cv2.imshow("YOLO26n-Face Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()