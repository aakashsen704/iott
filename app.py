from flask import Flask, render_template, Response
import cv2
import time
import os
import geocoder
from gpiozero import Buzzer
from datetime import datetime

app = Flask(__name__)

# Setup buzzer on GPIO pin 17
buzzer = Buzzer(17)

# Folder for saving snapshots
result_path = os.path.join('static', 'snapshots')
os.makedirs(result_path, exist_ok=True)

# Reading class labels
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLOv4-tiny model
net1 = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model1 = cv2.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Detection settings
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
last_save_time = 0
snapshot_index = 0

# Load camera
camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

def generate_frames():
    global frame_counter, last_save_time, snapshot_index
    starting_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_counter += 1
        classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)

        for (classid, score, box) in zip(classes, scores, boxes):
            if score >= 0.9:
                label = class_name[int(classid)]
                x, y, w, h = box
                recarea = w * h
                area = width * height

                if (recarea / area) <= 0.1 and y < 600:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, f"{round(score * 100, 2)}% {label}",
                                (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                    current_time = time.time()
                    if snapshot_index == 0 or (current_time - last_save_time) >= 2:
                        image_name = f"pothole{snapshot_index}.jpg"
                        txt_name = f"pothole{snapshot_index}.txt"
                        image_path = os.path.join(result_path, image_name)
                        txt_path = os.path.join(result_path, txt_name)

                        cv2.imwrite(image_path, frame)
                       
                        g = geocoder.ip('me')
                        coords = g.latlng if g.latlng else [0.0, 0.0]
                        with open(txt_path, 'w') as f:
                            f.write(str(coords))
                        last_save_time = current_time
                        snapshot_index += 1

                        buzzer.on()
                        time.sleep(0.2)
                        buzzer.off()

        elapsed_time = time.time() - starting_time
        fps = frame_counter / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    logs = []

    for filename in os.listdir(result_path):
        if filename.endswith(".jpg"):
            txt_name = filename.replace(".jpg", ".txt")
            txt_path = os.path.join(result_path, txt_name)
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    coords = f.read().strip()

                # Ensure coordinates are valid (non-empty and comma-separated lat/lng)
                if coords and ',' in coords:
                    timestamp = datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(result_path, filename))
                    ).strftime('%Y-%m-%d %H:%M:%S')
                    logs.append({
                        'image': filename,
                        'coords': coords,
                        'timestamp': timestamp
                    })

    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('index.html', logs=logs)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
