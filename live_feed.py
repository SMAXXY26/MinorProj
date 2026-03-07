import cv2
from ultralytics import YOLO

def run_live_feed(model_path="runs/detect/runs/gun_model8/weights/best.pt"):
    """
    Runs YOLOv8 inference on a live webcam feed using custom trained weights.
    """
    print(f"Loading model from: {model_path}")
    try:
        # Load your trained YOLOv8 model
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model and the weights path is correct.")
        return

    # Open the default camera (0). Use 1 or 2 if you have multiple cameras.
    # If you have an IP camera or stream URL, replace 0 with the URL string.
    print("Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        
        if not success:
            print("Failed to grab frame. Exiting...")
            break

        # Run YOLO inference on the frame
        # conf=0.5 sets a confidence threshold of 50%
        results = model(frame, conf=0.5)
        
        # Plot the predictions (bounding boxes and labels) on the frame
        # results[0].plot() returns a numpy array representing the annotated image
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("Live Feed - Object Detection", annotated_frame)
        
        # Wait for 1 ms and check if the 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_feed()
