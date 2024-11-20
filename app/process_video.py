import os
import cv2
from ultralytics import YOLO
import numpy as np
import csv
import re
import datetime

# ----------------------------
# Configurações Iniciais
# ----------------------------

# Caminhos para os modelos e arquivos
VEHICLE_MODEL_PATH = 'data/yolov_models/yolov8n.pt'
PLATE_CASCADE_PATH = 'utils/haarcascade_russian_plate_number.xml'
PLATE_RECOGNITION_MODEL_PATH = 'data/yolov_models/best_pre.pt'
OUTPUT_CSV_PATH = 'plates_detected_jaime.csv'

# Parâmetros
VEHICLE_CONF_THRESHOLD = 0.5

# Para armazenar o último horário em que cada placa foi detectada
last_seen_plates = {}

# ----------------------------
# Funções Auxiliares
# ----------------------------

def save_plate_info(plate, bounding_box, timestamp, output_csv=OUTPUT_CSV_PATH):
    """
    Salva apenas a última ocorrência de uma placa no CSV.
    """
    global last_seen_plates

    # Se a placa já foi vista antes e o timestamp é igual, não adiciona
    if plate in last_seen_plates and last_seen_plates[plate] == timestamp:
        return

    # Atualiza o registro mais recente da placa
    last_seen_plates[plate] = timestamp

    # Salva no CSV
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([plate, bounding_box, timestamp])
    print(f"Placa {plate} registrada no CSV às {timestamp}")

def is_valid_plate(plate):
    """
    Valida o formato da placa.
    """
    pattern = r'^[A-Z]{3}\d{4}$|^[A-Z]{3}\d[A-Z]\d{2}$'
    return re.match(pattern, plate) is not None

def detect_vehicles(frame, model, conf_threshold=VEHICLE_CONF_THRESHOLD):
    """
    Detecta veículos no frame.
    """
    results = model(frame, conf=conf_threshold)
    vehicle_boxes = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            if class_id in [2, 3, 5, 7] and confidence >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return frame, vehicle_boxes

def recognize_plate_characters(plate_roi, recognition_model):
    """
    Reconhece os caracteres na região de interesse da placa.
    """
    results = recognition_model(plate_roi)
    result = results[0]
    detected_characters = []

    for box in result.boxes:
        x1, character = int(box.xyxy[0][0]), recognition_model.names[int(box.cls[0])]
        detected_characters.append((x1, character))

    detected_characters = sorted(detected_characters, key=lambda x: x[0])
    plate_text = ''.join([char for _, char in detected_characters])
    return plate_text

def detect_and_recognize_plate(frame, plate_cascade, vehicle_boxes, plate_recognition_model):
    """
    Detecta e reconhece placas.
    """
    plates_info = []

    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        vehicle_roi = frame[y1:y2, x1:x2]

        if vehicle_roi.size == 0:
            continue

        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        detected_plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        for (px, py, pw, ph) in detected_plates:
            plate_roi = vehicle_roi[py:py + ph, px:px + pw]
            plate_text = recognize_plate_characters(plate_roi, plate_recognition_model)

            if is_valid_plate(plate_text):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bounding_box = f"({x1},{y1},{x2},{y2})"
                plates_info.append((plate_text, bounding_box, timestamp))
    
    return plates_info

# ----------------------------
# Função Principal
# ----------------------------

def live_video_capture(vehicle_model, plate_cascade, plate_recognition_model):
    """
    Realiza a captura de vídeo e processa os frames ao vivo.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    # Inicializar o CSV com cabeçalho, se necessário
    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Plate_Number', 'Bounding_Box', 'Timestamp'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, vehicle_boxes = detect_vehicles(frame, vehicle_model)

        plates_info = detect_and_recognize_plate(frame, plate_cascade, vehicle_boxes, plate_recognition_model)
        for plate, bounding_box, timestamp in plates_info:
            save_plate_info(plate, bounding_box, timestamp)

        cv2.imshow('Live Video Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    if not os.path.exists(VEHICLE_MODEL_PATH):
        print(f"Arquivo do modelo {VEHICLE_MODEL_PATH} não encontrado.")
    elif not os.path.exists(PLATE_CASCADE_PATH):
        print(f"Arquivo do Haar Cascade {PLATE_CASCADE_PATH} não encontrado.")
    elif not os.path.exists(PLATE_RECOGNITION_MODEL_PATH):
        print(f"Arquivo do modelo de reconhecimento {PLATE_RECOGNITION_MODEL_PATH} não encontrado.")
    else:
        vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        plate_recognition_model = YOLO(PLATE_RECOGNITION_MODEL_PATH)
        plate_cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)

        live_video_capture(vehicle_model, plate_cascade, plate_recognition_model)
