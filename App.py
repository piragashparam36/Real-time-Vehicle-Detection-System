import cv2
from ultralytics import YOLO
import os
# from google.colab.patches import cv2_imshow # Import the cv2_imshow function from google.colab.patches

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the small, pre-trained model (yolov8n.pt)

# Open the video file
video_path = r'C:\Users\pirag\Documents\Projects\VD\Vehicles.mp4' 
cap = cv2.VideoCapture(video_path)

# Get video details for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_detections.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the current frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detections

    # Show the frame (optional)
    # cv2_imshow(annotated_frame) # Use cv2_imshow instead of cv2.imshow

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Exit the video on key press (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()