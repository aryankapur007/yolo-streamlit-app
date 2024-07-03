import argparse
import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import streamlit as st

def predict_image(image_path, model):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform prediction
    results = model(img)

    # Display predictions
    annotated_img = img.copy()  # Make a copy of the original image
    st.write(type(results))
    # Iterate over each prediction in results (assuming results is a list)
    for bbox in results.boxes.xyxy:
        x1, y1, x2, y2 = bbox[:4]  # Extract coordinates
        label = results.names[bbox[5]]  # Get class label using class index
        confidence = bbox[4]  # Confidence score

    # Draw bounding box and label on the original image
        cv2.rectangle(results.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img.jpg, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    annotated_image_path = 'annotated_image.jpg'
    cv2.imwrite(annotated_image_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    return annotated_image_path