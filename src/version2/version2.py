import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model (You can use a pretrained model or a custom one)
model = YOLO('3custom_trained_model.pt')  # Replace with your custom YOLOv8 model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of all possible classes in YOLOv8
ALL_CLASSES = model.names  # This will give you a dictionary of class names

# Directory to save last seen images
SAVE_DIR = "last_seen_images"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# File to store user feedback
FEEDBACK_FILE = "user_feedback.txt"


def blur_faces(frame):
    """Detects and blurs faces in the given frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = frame[y:y+h, x:x+w]
        # Apply a Gaussian blur to the face region
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        # Place the blurred face back onto the frame
        frame[y:y+h, x:x+w] = blurred_face

    return frame


def save_last_seen_image(image, object_name):
    """Save the last seen image of the object locally, overwriting the old one."""
    image_path = os.path.join(SAVE_DIR, f"{object_name}.jpg")
    cv2.imwrite(image_path, image)  # Save the image
    return image_path

def detect_objects_in_video(frame_placeholder, last_seen_placeholder, selected_classes, target_class, video_file):
    # Check if a video file is provided
    if not video_file:
        st.error("Please upload a video file.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        st.error("Unable to open the video file.")
        return

    # Get video frame rate (FPS) and calculate the interval
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps * 10)  # Number of frames to skip for 10 seconds

    frame_count = 0  # Keep track of the frame number
    last_seen_image = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame only if it's at the specified interval
        if frame_count % frame_interval == 0:
            # Blur faces for privacy
            frame = blur_faces(frame)

            # Run YOLOv8 object detection
            results = model(frame)  # Get predictions for the frame

            # Filter detections by the selected classes
            if selected_classes:
                # Convert selected class names to class indices
                selected_class_indices = [list(ALL_CLASSES.values()).index(cls) for cls in selected_classes]

                # Filter detections based on selected class indices
                filtered_results = [det for det in results[0].boxes if int(det.cls) in selected_class_indices]
                results[0].boxes = filtered_results

            # Annotate the frame with the filtered detections
            annotated_frame = results[0].plot()  # Annotate the frame with detections

            # Check if the target class is in the detected objects
            detected_classes = [ALL_CLASSES[int(det.cls)] for det in results[0].boxes]
            if target_class in detected_classes:
                # Convert annotated frame to RGB and store it as the last seen frame of the target class
                last_seen_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Save the last seen image locally
                save_last_seen_image(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), target_class)

            # Convert the frame from BGR to RGB for live video display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the video feed and last seen snapshot side by side
            frame_placeholder.image(annotated_frame_rgb, caption="Video Feed", use_column_width=True)
            if last_seen_image is not None:
                last_seen_placeholder.image(last_seen_image, caption=f"Last Seen '{target_class}'", use_column_width=True)

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    cap.release()

def collect_feedback(target_class):
    """Collects user feedback and saves it to a text file."""
    st.subheader("User Feedback")
    feedback = st.text_area("Provide your feedback about the detection results:")

    if st.button("Submit Feedback"):
        if feedback.strip():  # Ensure feedback is not empty
            with open(FEEDBACK_FILE, "a") as file:
                file.write(f"Target Class: {target_class}\n")
                file.write(f"Feedback: {feedback}\n")
                file.write("-" * 50 + "\n")
            st.success("Thank you for your feedback! It has been saved.")
        else:
            st.error("Feedback cannot be empty. Please provide your comments.")



def main():
    st.title("REACH - Video Input")

    # Video file uploader
    video_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])

    # Dropdown to select classes for general detection
    selected_classes = st.multiselect(
        "Select Classes to Detect (Leave empty for all classes)",
        options=list(ALL_CLASSES.values()),  # All class names
        default=[]  # Default is no classes selected (detect all)
    )

    # Input text for specific object tracking
    target_class = st.text_input("Enter Object Class to Track (e.g., 'cell phone')")

    

    # Create side-by-side columns for displaying the video feed and last seen frame
    col1, col2 = st.columns(2)
    with col1:
        frame_placeholder = st.empty()  # Placeholder for live video feed
    with col2:
        last_seen_placeholder = st.empty()  # Placeholder for last seen snapshot

    # Start detection button
    if st.button("Start Video Detection"):
        if video_file:
            # Save the uploaded file to a temporary location
            temp_video_path = os.path.join("temp_video", video_file.name)
            os.makedirs("temp_video", exist_ok=True)
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())

            # Run the detection on the video file
            detect_objects_in_video(frame_placeholder, last_seen_placeholder, selected_classes, target_class, temp_video_path)
        else:
            st.error("Please upload a video file to start detection.")

    # Add feedback section
    if target_class:
        collect_feedback(target_class)


if __name__ == "__main__":
    main()
