import os
import cv2
from ultralytics import YOLO
import csv

# ----------------------------
# Configurações Iniciais
# ----------------------------

# Caminhos para os modelos e arquivos
VEHICLE_MODEL_PATH = 'data/yolov_models/yolov8n.pt'  # Modelo YOLOv8 para detecção de veículos
PLATE_CASCADE_PATH = 'utils/haarcascade_russian_plate_number.xml'  # Haar Cascade para detecção de placas
PLATE_RECOGNITION_MODEL_PATH = 'data/yolov_models/best_pre.pt'  # Modelo YOLOv8 treinado para reconhecimento de caracteres das placas
OUTPUT_CSV_PATH = 'plates_detected_jaime.csv'  # Caminho para salvar as informações das placas
PLATES_IMAGES_DIR = 'detected_plates_jaime'  # Diretório para salvar as imagens das placas detectadas

# Parâmetros
VEHICLE_CONF_THRESHOLD = 0.5  # Confiança mínima para detecção de veículos
ZOOM_SCALE = 2  # Fator de zoom para a placa

# ----------------------------
# Funções Auxiliares
# ----------------------------

def save_plates_info(plates_info, output_csv=OUTPUT_CSV_PATH):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'X1', 'Y1', 'X2', 'Y2', 'Plate_Number'])
        for info in plates_info:
            writer.writerow(info)

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
    
    if not os.path.exists(PLATES_IMAGES_DIR):
        os.makedirs(PLATES_IMAGES_DIR)
    
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
            
            # Ampliar a imagem da placa para maior clareza
            zoomed_plate = cv2.resize(plate_roi, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_LINEAR)
            
            # Salvar a imagem ampliada da placa
            plate_image_filename = f"frame_{frame_number}_plate_{idx + 1}.jpg"
            plate_image_path = os.path.join(PLATES_IMAGES_DIR, plate_image_filename)
            cv2.imwrite(plate_image_path, zoomed_plate)
            
            # Reconhecer os caracteres na placa ampliada
            plate_text = recognize_plate_characters(zoomed_plate, plate_recognition_model)
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
    plates_info_all = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_with_vehicles, vehicle_boxes = detect_vehicles(frame.copy(), vehicle_model, conf_threshold)
        frame_with_plates, plates_info = detect_and_recognize_plate(frame_with_vehicles, plate_cascade, vehicle_boxes, plate_recognition_model, frame_count)
        plates_info_all.extend(plates_info)
        
        cv2.imshow('Live Video Feed', frame_with_plates)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    save_plates_info(plates_info_all, OUTPUT_CSV_PATH)
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
