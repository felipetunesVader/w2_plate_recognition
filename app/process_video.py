import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import csv

# ----------------------------
# Configurações Iniciais
# ----------------------------

# Caminhos para os modelos e arquivos
VEHICLE_MODEL_PATH = 'data/yolov_models/yolov8n.pt' # Modelo YOLOv8 para detecção de veículos
PLATE_CASCADE_PATH = 'utils/haarcascade_russian_plate_number.xml'  # Haar Cascade para detecção de placas
PLATE_RECOGNITION_MODEL_PATH = 'data/yolov_models/best_pre.pt'  # Modelo YOLOv8 treinado para reconhecimento de placas
INPUT_VIDEO_PATH = 'data/images/brazil.mp4'  # Caminho para o vídeo de entrada
OUTPUT_VIDEO_PATH = 'output_video_detectedJAIME.mp4'  # Caminho para salvar o vídeo processado
OUTPUT_CSV_PATH = 'plates_detected_jaime.csv'  # Caminho para salvar as informações das placas
PLATES_IMAGES_DIR = 'detected_plates_jaime'  # Diretório para salvar as imagens das placas detectadas

# Parâmetros
VEHICLE_CONF_THRESHOLD = 0.5  # Confiança mínima para detecção de veículos
ZOOM_SCALE = 2  # Fator de zoom para a placa

# Mapeamento de classes para caracteres reconhecidos
CLASS_MAP = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z'
}

# ----------------------------
# Funções Auxiliares
# ----------------------------

def save_plates_info(plates_info, output_csv=OUTPUT_CSV_PATH):
    """
    Salva as informações das placas detectadas em um arquivo CSV.
    
    Args:
        plates_info (list): Lista de tuplas com informações das placas (frame, coordenadas e texto).
        output_csv (str): Nome do arquivo CSV para salvar as informações.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'X1', 'Y1', 'X2', 'Y2', 'Plate_Number'])
        for info in plates_info:
            frame_number, x1, y1, x2, y2, plate_text = info
            writer.writerow([frame_number, x1, y1, x2, y2, plate_text])

def sort_boxes_left_to_right(boxes):
    """
    Ordena uma lista de caixas delimitadoras da esquerda para a direita.
    
    Args:
        boxes (list): Lista de caixas delimitadoras [(x1, y1, x2, y2), ...]
    
    Returns:
        sorted_boxes (list): Lista de caixas ordenadas da esquerda para a direita.
    """
    return sorted(boxes, key=lambda box: box[0])

# ----------------------------
# Funções de Detecção e Reconhecimento
# ----------------------------

def detect_vehicles(frame, model, conf_threshold=VEHICLE_CONF_THRESHOLD):
    """
    Detecta veículos em um frame usando o modelo YOLOv8.
    
    Args:
        frame (numpy.ndarray): Frame da imagem.
        model (YOLO): Modelo YOLOv8 carregado.
        conf_threshold (float): Confiança mínima para detecção.
    
    Returns:
        frame (numpy.ndarray): Frame com caixas delimitadoras desenhadas.
        vehicle_boxes (list): Lista de caixas delimitadoras dos veículos.
    """
    results = model(frame, conf=conf_threshold)
    vehicle_boxes = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            # Classes de veículos no COCO: carro (2), motocicleta (3), ônibus (5), caminhão (7)
            if class_id in [2, 3, 5, 7] and confidence >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                vehicle_boxes.append((x1, y1, x2, y2))
                # Desenhar a caixa no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{result.names[class_id]} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0), 2)
    
    return frame, vehicle_boxes

def detect_and_recognize_plate(frame, plate_cascade, vehicle_boxes, plate_recognition_model, frame_number, plates_images_dir=PLATES_IMAGES_DIR):
    """
    Detecta placas de licença dentro das regiões dos veículos detectados e reconhece os caracteres.
    
    Args:
        frame (numpy.ndarray): Frame da imagem.
        plate_cascade (cv2.CascadeClassifier): Haar Cascade para placas de licença.
        vehicle_boxes (list): Lista de caixas delimitadoras dos veículos.
        plate_recognition_model (YOLO): Modelo treinado para reconhecimento de placas.
        frame_number (int): Número do frame atual.
        plates_images_dir (str): Diretório para salvar as imagens das placas detectadas.
    
    Returns:
        frame (numpy.ndarray): Frame com caixas e textos desenhados.
        plates_info (list): Lista de tuplas com informações das placas (frame, coordenadas e texto).
    """
    plates_info = []
    
    # Assegurar que o diretório para salvar as placas existe
    if not os.path.exists(plates_images_dir):
        os.makedirs(plates_images_dir)
    
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            continue  # Evitar erros se a ROI estiver vazia
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Detectar placas de licença
        detected_plates = plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        
        for idx, (px, py, pw, ph) in enumerate(detected_plates):
            # Ajustar as coordenadas para o frame original
            px1_abs = x1 + px
            py1_abs = y1 + py
            px2_abs = px1_abs + pw
            py2_abs = py1_abs + ph
            plates_info.append((frame_number, px1_abs, py1_abs, px2_abs, py2_abs, ""))
            
            # Desenhar a caixa da placa no frame
            cv2.rectangle(frame, (px1_abs, py1_abs), (px2_abs, py2_abs), (0, 255, 0), 2)
            label = "Plate"
            cv2.putText(frame, label, (px1_abs, py1_abs - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
            
            # Extrair a ROI da placa para reconhecimento
            plate_roi = vehicle_roi[py:py + ph, px:px + pw]
            
            # Preparar a imagem para o modelo de reconhecimento
            # Supondo que o modelo espera uma imagem RGB
            plate_input = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)
            
            # Realizar a inferência com o modelo treinado
            recognition_results = plate_recognition_model(plate_input)
            
            # Extrair o texto reconhecido
            plate_text = ""
            for result in recognition_results:
                for box_recog in result.boxes:
                    class_id = int(box_recog.cls[0].item())  # Converter Tensor para int
                    character = CLASS_MAP.get(class_id, '')  # Obter o caractere correspondente
                    plate_text += character  # Concatenar o caractere
            
            # Atualizar a informação da placa
            plates_info[-1] = (frame_number, px1_abs, py1_abs, px2_abs, py2_abs, plate_text)
            
            # Desenhar o texto reconhecido na tela
            if plate_text:
                cv2.putText(frame, plate_text, (px1_abs, py2_abs + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Aplicar zoom na área da placa
                zoomed_plate = cv2.resize(plate_roi, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, 
                                          interpolation=cv2.INTER_LINEAR)
                
                # Definir a posição do zoom (canto superior direito com uma margem)
                frame_height, frame_width, _ = frame.shape
                zoom_height, zoom_width, _ = zoomed_plate.shape
                
                x_offset = frame_width - zoom_width - 10
                y_offset = 10
                
                # Verificar se o zoom cabe na imagem
                if x_offset + zoom_width <= frame_width and y_offset + zoom_height <= frame_height:
                    frame[y_offset:y_offset + zoom_height, x_offset:x_offset + zoom_width] = zoomed_plate
                
                # Salvar a imagem da placa detectada na pasta
                plate_image_filename = f"frame_{frame_number}_plate_{idx + 1}.jpg"
                plate_image_path = os.path.join(plates_images_dir, plate_image_filename)
                cv2.imwrite(plate_image_path, plate_roi)
    
    return frame, plates_info

# ----------------------------
# Função Principal para Processar o Vídeo
# ----------------------------

def process_video(input_video_path, output_video_path, vehicle_model, plate_cascade, plate_recognition_model, conf_threshold=VEHICLE_CONF_THRESHOLD):
    """
    Processa um vídeo para detectar veículos e placas de licença, reconhecer os caracteres das placas, 
    aplicar zoom nas placas, exibir os números na tela e salvar as imagens das placas detectadas.
    
    Args:
        input_video_path (str): Caminho para o vídeo de entrada.
        output_video_path (str): Caminho para salvar o vídeo processado.
        vehicle_model (YOLO): Modelo YOLOv8 para detecção de veículos.
        plate_cascade (cv2.CascadeClassifier): Haar Cascade para placas de licença.
        plate_recognition_model (YOLO): Modelo treinado para reconhecimento de placas.
        conf_threshold (float): Confiança mínima para detecção de veículos.
    """
    # Abrir o vídeo de entrada
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {input_video_path}")
        return
    
    # Obter propriedades do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Largura do frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Altura do frame
    
    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    plates_info_all = []  # Para armazenar todas as informações das placas
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processando frame {frame_count}/{total_frames}", end='\r')
        
        # Detectar veículos
        frame_with_vehicles, vehicle_boxes = detect_vehicles(frame.copy(), vehicle_model, conf_threshold)
        
        # Detectar placas de licença e reconhecer caracteres, salvar as imagens das placas
        frame_with_plates, plates_info = detect_and_recognize_plate(
            frame_with_vehicles, plate_cascade, vehicle_boxes, plate_recognition_model, frame_count
        )
        
        plates_info_all.extend(plates_info)  # Salvar as informações
        
        # Escrever o frame processado no vídeo de saída
        out.write(frame_with_plates)
    
    # Liberar os objetos
    cap.release()
    out.release()
    print("\nProcessamento concluído.")
    
    # Salvar as informações das placas em um CSV
    save_plates_info(plates_info_all, OUTPUT_CSV_PATH)
    print(f"Informações das placas salvas em {OUTPUT_CSV_PATH}")
    print(f"Imagens das placas salvas na pasta '{PLATES_IMAGES_DIR}'.")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    # Verificar se os arquivos necessários existem
    missing_files = []
    if not os.path.exists(VEHICLE_MODEL_PATH):
        missing_files.append(VEHICLE_MODEL_PATH)
    if not os.path.exists(PLATE_CASCADE_PATH):
        missing_files.append(PLATE_CASCADE_PATH)
    if not os.path.exists(PLATE_RECOGNITION_MODEL_PATH):
        missing_files.append(PLATE_RECOGNITION_MODEL_PATH)
    if not os.path.exists(INPUT_VIDEO_PATH):
        missing_files.append(INPUT_VIDEO_PATH)
    
    if missing_files:
        print("Os seguintes arquivos estão faltando:")
        for file in missing_files:
            print(f"- {file}")
            #print(VEHICLE_MODEL_PATH)
        print("Por favor, certifique-se de que todos os arquivos necessários estão presentes.")
    else:
        # Carregar os modelos
        print("Carregando modelos...")
        vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        plate_recognition_model = YOLO(PLATE_RECOGNITION_MODEL_PATH)
        plate_cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)
        
        # Processar o vídeo
        print("Iniciando o processamento do vídeo...")
        process_video(
            INPUT_VIDEO_PATH,
            OUTPUT_VIDEO_PATH,
            vehicle_model,
            plate_cascade,
            plate_recognition_model,
            VEHICLE_CONF_THRESHOLD
        )
        
        # Informar onde o vídeo foi salvo
        print(f"Vídeo processado salvo em: {OUTPUT_VIDEO_PATH}")
