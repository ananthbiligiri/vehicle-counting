import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

# Vehicle classes to count
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']

# Streamlit UI
st.title("🚗 Vehicle Detection & Counting using YOLOv8")
st.sidebar.header("Upload or Use Webcam")

# Option to upload a video
video_source = st.sidebar.radio("Choose input source:", ("Webcam", "Upload Video"))

vehicle_count_placeholder = st.empty()

# Create video display area
frame_display = st.empty()

def count_vehicles(results):
    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in vehicle_classes:
                count += 1
    return count

def process_frame(frame):
    results = model(frame)
    count = count_vehicles(results)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame, count

# Run detection
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, count = process_frame(frame)
            vehicle_count_placeholder.markdown(f"### Vehicles Detected: {count}")

            frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
else:
    cap = cv2.VideoCapture(0)

    st.warning("Press 'Stop' to release the webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, count = process_frame(frame)
        vehicle_count_placeholder.markdown(f"### Vehicles Detected: {count}")

        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(0.01)

    cap.release()
