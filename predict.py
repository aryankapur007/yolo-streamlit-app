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
    results = model.predict(img)

    # Display predictions
    annotated_img = img.copy()  # Make a copy of the original image
    st.write(type(results))

    boxes = results[0].boxes
    res_plotted = results[0].plot()[:, :, ::-1]

    st.image(
         res_plotted, caption="Detected Image", use_column_width=True
)

    #Iterate over each prediction in results (assuming results is a list)
    for result in results:
        bbox = result['bbox']
        x1, y1, x2, y2 = bbox[:4]  # Extract coordinates
        label = result['class']  # Get class label using class index
        confidence = result['confidence']  # Confidence score

    #     # Draw bounding box and label on the original image
        cv2.rectangle(annotated_img, (int(x1), int(y1)),
                       (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated_img, f'{label} {confidence:.2f}', (int(
             x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    annotated_image_path = 'annotated_image.jpg'
    cv2.imwrite(annotated_image_path, cv2.cvtColor(
        res_plotted, cv2.COLOR_RGB2BGR))

    return annotated_image_path