#!/bin/bash
# train_model.sh
# Script to train YOLOv8 model for weapon detection

# Navigate to the correct directory
cd /home/vinesh/ML/MinorProj || exit

# Start training the model using the standard YOLO script (updated to 80 epochs)
echo "Starting YOLOv8 training on the weapon detection dataset (80 epochs)..."

# Run it in the background using nohup so training continues even if the terminal closes
nohup python3 "Train data.py" > training_output.log 2>&1 &
PID=$!

echo "Training launched in the background with PID $PID."
echo "You can view the logs in real-time by running:"
echo "tail -f /home/vinesh/ML/MinorProj/training_output.log"
