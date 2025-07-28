import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# Load YOLOv8 model
model = YOLO('C:/Users/HP/OneDrive/Desktop/LOGO3/runs/detect/train2/weights/best.pt')  # Adjust this if needed

# App title
st.set_page_config(page_title="Logo Detection App", layout="centered")
st.title("🔍 Logo Detection App")
st.markdown("Upload an image or use your webcam to detect logos using your trained YOLOv8 model.")

# Sidebar options
st.sidebar.title(" Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
source_type = st.sidebar.radio("Select Input Source", ["Image Upload", "Webcam"])

# Process image upload
def detect_on_image(uploaded_img):
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_path = tmp.name
        uploaded_img.save(img_path)

    # Run YOLO inference
    results = model(img_path, conf=confidence_threshold)
    result_img = results[0].plot()
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Show result
    st.image(result_rgb, caption=" Detected Logos", use_column_width=True)

    # Save option
    if st.button("Save Result"):
        save_path = "detection_result.jpg"
        cv2.imwrite(save_path, result_img)
        st.success(f"Saved result to `{save_path}`")

    # Show class info
    detections = results[0].boxes
    if detections:
        st.markdown("### 🧾 Detections:")
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            st.write(f"- **{class_name}** with confidence {conf:.2f}")
    else:
        st.warning("No logos detected.")

# Webcam logic
def detect_from_webcam():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    st.info("📸 Press 'Stop Webcam' to end.")
    stop_button = st.button("Stop Webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Waiting for webcam...")
            continue

        results = model.predict(frame, conf=confidence_threshold)
        result_frame = results[0].plot()
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        stframe.image(result_rgb, channels="RGB", use_column_width=True)

        if stop_button:
            break

    cap.release()
    st.success(" Webcam stopped.")

# Main logic
if source_type == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Uploaded Image", use_column_width=True)
        detect_on_image(img)

elif source_type == "Webcam":
    detect_from_webcam()
