import cv2
import mediapipe as mp
import time

# Aliases for easier access
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store results
latest_result = None

# Callback function to receive results asynchronously
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# 1. Configure the landmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

# 2. Open Webcam
cap = cv2.VideoCapture(0)

# Use 'with' to ensure the landmarker is properly closed
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord("q"): break

        # Prepare frame: Convert BGR to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 3. Detect (send to async processor)
        # We must provide a monotonically increasing timestamp in ms
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # 4. Draw results if available
        if latest_result and latest_result.pose_landmarks:
            for pose_landmarks in latest_result.pose_landmarks:
                # Custom drawing logic (since drawing_utils is legacy)
                for landmark in pose_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        cv2.imshow('MediaPipe Tasks API - Pose', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()