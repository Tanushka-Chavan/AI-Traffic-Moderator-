# Import necessary modules
from flask import Flask, render_template, Response, jsonify  # Flask modules for web app and video streaming
import cv2  # OpenCV for image processing
from ultralytics import YOLO  # YOLO for object detection
import time  # Time module to manage traffic signal timing

# Initialize the Flask app
app = Flask(__name__)

# Load YOLOv8n model from the given path
model = YOLO("models/yolov8n.pt")

# Open the default camera (0 is typically the laptop camera or first available)
cap = cv2.VideoCapture(0)

# Reduce the buffer size to minimize frame delay (latency)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Define vehicle classes to be considered from YOLO classes (based on COCO dataset)
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Initialize traffic signal state and timers
traffic_signal = "red"  # Initial signal color
signal_timer = 15       # Time in seconds to hold the current signal
last_signal_change = time.time()  # Timestamp of the last signal change

# Function to detect vehicles in a frame
def detect_vehicles(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for YOLO model
    results = model(rgb_frame)[0]  # Run YOLO model and extract first result
    vehicles = []  # List to hold detected vehicle data

    # Iterate through detected boxes
    for box in results.boxes:
        class_id = int(box.cls[0])  # Get the class ID of the detected object
        if class_id in VEHICLE_CLASSES:  # If the object is a vehicle
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            vehicles.append((class_id, (x1, y1, x2, y2)))  # Append vehicle data to list

    return vehicles  # Return list of detected vehicles

# Generator function to process and stream video frames
def process_frame():
    global traffic_signal, signal_timer, last_signal_change

    while True:
        cap.grab()  # Grab the latest frame without decoding to minimize latency
        ret, frame = cap.read()  # Read the frame
        if not ret:
            break  # If reading fails, exit the loop

        vehicles = detect_vehicles(frame)  # Detect vehicles in the current frame
        vehicle_count = len(vehicles)  # Count the number of detected vehicles

        # Draw bounding boxes and labels on detected vehicles
        for class_id, bbox in vehicles:
            x1, y1, x2, y2 = bbox
            label = VEHICLE_CLASSES[class_id]
            color = (0, 255, 0)  # Green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Draw label

        # Traffic signal timing and logic
        current_time = time.time()
        elapsed_time = current_time - last_signal_change  # Time since last signal change

        # If it's time to change the signal
        if elapsed_time >= signal_timer:
            if traffic_signal == "red":
                # Switch to green/yellow depending on traffic volume
                traffic_signal = "green" if vehicle_count >= 10 else "yellow" if vehicle_count >= 5 else "red"
                signal_timer = 15 if vehicle_count >= 10 else 10 if vehicle_count >= 5 else 15
            elif traffic_signal == "green":
                traffic_signal = "yellow"  # After green, switch to yellow
                signal_timer = 4
            elif traffic_signal == "yellow":
                traffic_signal = "red"  # After yellow, switch to red
                signal_timer = 15
            last_signal_change = current_time  # Update the timestamp of last signal change

        # Display signal and vehicle count on the frame
        cv2.putText(frame, f"Signal: {traffic_signal}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode the frame as JPEG to send it over HTTP
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Compress JPEG slightly for speed

        # Yield the frame in multipart format for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Route for main page
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML frontend

# Route to serve video stream
@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')  # MJPEG stream

# Route to get current traffic status as JSON (used for frontend updates)
@app.route('/traffic_status')
def traffic_status():
    return jsonify({"traffic_light": traffic_signal, "vehicle_count": len(detect_vehicles(cap.read()[1]))})

# Main function to run the Flask app
if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)  # Run app with threading enabled for better performance
    finally:
        cap.release()  # Release the camera on exit
        cv2.destroyAllWindows()  # Close any OpenCV windows
