from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("best.pt")

# Initialize the camera
camera = cv2.VideoCapture(2)
img_counter = 0

while True:
    ret, frame = camera.read()

    if not ret:
        break

    # Perform inference
    results = model(frame, device='cuda')  # Perform inference using the model

    # Draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            print("Box coordinates (xyxy):", box.xyxy)

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Ensure this is correct format
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optional: Print confidence score for each box
            print("Confidence score:", box.conf)

    # Display the frame with bounding boxes
    cv2.imshow("test", frame)

    # Save the frame with bounding boxes
    img_path = f"path/opencv_frame_{img_counter}.png"
    cv2.imwrite(img_path, frame)
    img_counter += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()