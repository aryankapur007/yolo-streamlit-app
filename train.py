from ultralytics import YOLO
import time

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Record the start time
start_time = time.time()

# Train the model
results = model.train(
    data='C:/cogentinfo/yolo-learning/yolo-streamlit-app/Cars Detection/data.yaml',  # Correct path to the data.yaml file
    epochs=3,
    imgsz=640
)

# Record the end time
end_time = time.time()

# Calculate and print the training time in minutes
training_time_seconds = end_time - start_time
training_time_minutes = training_time_seconds / 60

print(f"Training time: {training_time_minutes:.2f} minutes")
