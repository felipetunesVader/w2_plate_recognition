import os
import cv2
from ultralytics import YOLO
import numpy as np
import csv
import time
from collections import Counter
import re

# ----------------------------
# Configurações Iniciais
# ----------------------------

# Caminhos para os modelos e arquivos
VEHICLE_MODEL_PATH = 'data/yolov_models/yolov8n.pt'  # Modelo YOLOv8 para detecção de veículos
PLATE_CASCADE_PATH = 'utils/haarcascade_russian_plate_number.xml'  # Haar Cascade para detecção de placas
PLATE_RECOGNITION_MODEL_PATH = 'data/yolov_models/best_pre.pt'  # Modelo YOLOv8 treinado para reconhecimento de caracteres das placas
OUTPUT_CSV_PATH = 'plates_detected_jaime.csv'  # Caminho para salvar as informações das placas
PLATES_IMAGES_DIR = 'detected_plates_jaime'  # Diretório para salvar as imagens das placas detectadas (opcional)
RESULT_FOLDER = 'resultados_from_video_jaime'  # Diretório para salvar os resultados do reconhecimento

# Parâmetros
VEHICLE_CONF_THRESHOLD = 0.5  # Confiança mínima para detecção de veículos
ZOOM_SCALE = 2  # Fator de zoom para a placa

# Parâmetros de Estado e Tempo
DETECTION_TIMEOUT = 2  # Tempo em segundos para considerar que o veículo saiu da cena

# ----------------------------
# Funções Auxiliares
# ----------------------------

def save_plate_info(plate_info, output_csv=OUTPUT_CSV_PATH):
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(plate_info)

def is_valid_plate(plate):
    # Exemplo de regex para placas brasileiras (AAA-9999 ou AAA9A99)
    pattern = r'^[A-Z]{3}\d{4}$|^[A-Z]{3}\d[A-Z]\d{2}$'
    return re.match(pattern, plate) is not None

def detect_vehicles(frame, model, conf_threshold=VEHICLE_CONF_THRESHOLD):
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
                label = f"{result.names[class_id]} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame, vehicle_boxes

def recognize_plate_characters(plate_roi, recognition_model):
    # Fazer a inferência no ROI da placa
    results = recognition_model(plate_roi)
    result = results[0]
    detected_characters = []
    
    # Extrair os caracteres reconhecidos
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa
        class_id = int(box.cls[0].item())  # ID da classe (número ou letra detectada)
        character = recognition_model.names[class_id]  # Obter o caractere correspondente
        detected_characters.append((x1, character))  # Armazena coordenada X e caractere

    # Ordenar os caracteres da esquerda para a direita e formar o texto da placa
    detected_characters = sorted(detected_characters, key=lambda x: x[0])
    plate_text = ''.join([char for _, char in detected_characters])
    
    return plate_text

def detect_and_recognize_plate(frame, plate_cascade, vehicle_boxes, plate_recognition_model, frame_number):
    plates_info = []
    
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            continue
        
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        detected_plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        for idx, (px, py, pw, ph) in enumerate(detected_plates):
            px1_abs, py1_abs = x1 + px, y1 + py
            px2_abs, py2_abs = px1_abs + pw, py1_abs + ph
            plate_roi = vehicle_roi[py:py + ph, px:px + pw]
            
            # Reconhecer os caracteres na placa
            plate_text = recognize_plate_characters(plate_roi, plate_recognition_model)
            plates_info.append((frame_number, px1_abs, py1_abs, px2_abs, py2_abs, plate_text))
            
            # Desenhar a placa e o texto reconhecido na imagem
            cv2.rectangle(frame, (px1_abs, py1_abs), (px2_abs, py2_abs), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (px1_abs, py2_abs + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, plates_info

# ----------------------------
# Função Principal para Captura de Vídeo ao Vivo
# ----------------------------

def live_video_capture(vehicle_model, plate_cascade, plate_recognition_model, conf_threshold=VEHICLE_CONF_THRESHOLD):
    cap = cv2.VideoCapture(0)  # 0 indica a webcam padrão
    
    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return
    
    frame_count = 0
    plates_buffer = []
    vehicle_present = False
    last_detection_time = 0
    
    # Inicializar o CSV com cabeçalho se não existir
    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'X1', 'Y1', 'X2', 'Y2', 'Plate_Number'])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Detectar veículos
        frame_with_vehicles, vehicle_boxes = detect_vehicles(frame.copy(), vehicle_model, conf_threshold)
        
        if vehicle_boxes:
            # Veículo está presente
            if not vehicle_present:
                # Novo veículo detectado
                vehicle_present = True
                plates_buffer = []  # Resetar o buffer
                print("Veículo detectado. Iniciando coleta de placas.")
            
            last_detection_time = current_time
            
            # Detectar e reconhecer placas
            frame_with_plates, plates_info = detect_and_recognize_plate(frame_with_vehicles, plate_cascade, vehicle_boxes, plate_recognition_model, frame_count)
            plates_buffer.extend(plates_info)
        else:
            # Nenhum veículo detectado
            if vehicle_present and (current_time - last_detection_time) > DETECTION_TIMEOUT:
                # Veículo saiu da cena
                vehicle_present = False
                print("Veículo saiu da cena. Processando leituras de placas.")
                
                if plates_buffer:
                    # Agregar as leituras de placas
                    plate_texts = [info[5] for info in plates_buffer]
                    plate_counts = Counter(plate_texts)
                    
                    # Selecionar a placa mais comum
                    most_common_plate, count = plate_counts.most_common(1)[0]
                    
                    # Validar a placa
                    if is_valid_plate(most_common_plate):
                        # Obter as coordenadas da placa para registro (usando a primeira ocorrência)
                        for info in plates_buffer:
                            if info[5] == most_common_plate:
                                frame_num, x1, y1, x2, y2, plate = info
                                break
                        
                        # Salvar no CSV
                        save_plate_info([frame_num, x1, y1, x2, y2, most_common_plate])
                        print(f"Placa registrada: {most_common_plate} no frame {frame_num}")
                        
                        # Opcional: Salvar a imagem da placa
                        if not os.path.exists(PLATES_IMAGES_DIR):
                            os.makedirs(PLATES_IMAGES_DIR)
                        plate_image = frame[y1:y2, x1:x2]
                        plate_image_path = os.path.join(PLATES_IMAGES_DIR, f"{most_common_plate}_{frame_num}.jpg")
                        cv2.imwrite(plate_image_path, plate_image)
                    else:
                        print(f"Leitura de placa inválida: {most_common_plate}")
                
                plates_buffer = []  # Resetar o buffer
        
        # Exibir o frame com as detecções
        cv2.imshow('Live Video Feed', frame_with_vehicles if vehicle_present else frame_with_vehicles)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Informações das placas salvas em {OUTPUT_CSV_PATH}")
    print(f"Imagens das placas salvas na pasta '{PLATES_IMAGES_DIR}'.")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    missing_files = []
    if not os.path.exists(VEHICLE_MODEL_PATH):
        missing_files.append(VEHICLE_MODEL_PATH)
    if not os.path.exists(PLATE_CASCADE_PATH):
        missing_files.append(PLATE_CASCADE_PATH)
    if not os.path.exists(PLATE_RECOGNITION_MODEL_PATH):
        missing_files.append(PLATE_RECOGNITION_MODEL_PATH)
    
    if missing_files:
        print("Os seguintes arquivos estão faltando:")
        for file in missing_files:
            print(f"- {file}")
        print("Por favor, certifique-se de que todos os arquivos necessários estão presentes.")
    else:
        print("Carregando modelos...")
        vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        plate_recognition_model = YOLO(PLATE_RECOGNITION_MODEL_PATH)
        plate_cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)
        
        print("Iniciando a captura de vídeo ao vivo...")
        live_video_capture(vehicle_model, plate_cascade, plate_recognition_model, VEHICLE_CONF_THRESHOLD)
