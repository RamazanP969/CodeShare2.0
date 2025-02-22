import os
import cv2
import time
from flask import Flask, Response, request, jsonify
from ultralytics import YOLO
import threading

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Загрузка модели YOLO
captured_images = []  # Хранилище для снимков
new_images = 0  # Количество новых снимков
lock = threading.Lock()

# Классы для детекции (люди и машины)
DETECTION_CLASSES = {0: "Person", 2: "Car"}  # COCO: 0 - человек, 2 - машина


def save_frame(frame):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"static/captures/{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename, timestamp


def generate_frames(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Ошибка подключения к видеопотоку")
        return

    last_detection_time = 0
    detection_interval = 10  # Интервал между снимками (сек)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Детекция объектов
        results = model(frame)
        object_detected = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                if class_id in DETECTION_CLASSES and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{DETECTION_CLASSES[class_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    object_detected = True

        # Сохранение снимка, если обнаружен человек или машина
        if object_detected and time.time() - last_detection_time > detection_interval:
            last_detection_time = time.time()
            filename, timestamp = save_frame(frame)
            with lock:
                captured_images.append({"filename": filename, "timestamp": timestamp})
                global new_images
                new_images += 1

        # Кодирование кадра в JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Возврат кадра в формате MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    url = data.get("url")
    protocol = data.get("protocol")

    if not url or not protocol:
        return jsonify({"error": "URL или протокол не указан"}), 400

    return Response(generate_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_captures', methods=['GET'])
def get_captures():
    global new_images
    with lock:
        new_images = 0  # Сброс счетчика новых изображений
    return jsonify(captured_images)


@app.route('/new_images_count', methods=['GET'])
def new_images_count():
    with lock:
        return jsonify({"new_images": new_images})


if __name__ == '__main__':
    os.makedirs("static/captures", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
