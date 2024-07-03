import streamlit as st
from predict import predict_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from ultralytics import YOLO

model = YOLO('runs/detect/train7/weights/best.pt')

# Function to plot annotated image using Streamlit and Matplotlib
def plot_annotated_image(image_path):
    # Load image
    img = plt.imread(image_path)

    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Load annotations from the image (if available)
    # You might need to load bounding box coordinates and labels from a file or directly from the YOLO results

    # Example: Drawing a rectangle
    rect = patches.Rectangle((50, 100), 200, 400, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Example: Adding a label
    plt.text(50, 50, 'Car', fontsize=12, color='white', weight='bold', bbox=dict(facecolor='red', alpha=0.5))

    # Customize plot parameters
    ax.axis('off')  # Turn off axis
    st.pyplot(fig)  # Display the plot using Streamlit

# Main Streamlit app code

st.title('YOLOv8 Object Detection with Streamlit')

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    image_path = './uploaded_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.read())

        # Predict and annotate the uploaded image using predict_image function
    annotated_image_path=predict_image(image_path, model)

    # Display annotated image using Matplotlib and Streamlit
    plot_annotated_image(annotated_image_path)

    # Remove the temporary uploaded image
    os.remove(image_path)
