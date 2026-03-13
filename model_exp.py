from ultralytics import YOLO

model = YOLO("runs/detect/runs/gun_model3cls3/weights/best v1.pt")   # or yolov8n.pt
model.export(format="onnx", imgsz=416, half=True)

