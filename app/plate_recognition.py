import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Carregar o modelo YOLOv8 treinado para caracteres
model = YOLO('data/yolov_models/best_pre.pt')  # Substitua pelo caminho do seu modelo treinado

# Caminho da pasta de imagens
image_folder = 'detected_plates_jaime'  # Pasta com as imagens
output_folder = 'resultados_from_video_jaime'  # Pasta onde serão salvos os resultados

# Criar a pasta de resultados, se não existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Listar todas as imagens na pasta 'tes'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Loop para processar cada imagem
for idx, image_file in enumerate(image_files):
    # Caminho completo da imagem
    image_path = os.path.join(image_folder, image_file)

    # Fazer a inferência na imagem
    results = model(image_path)
    result = results[0]  # Acessar o primeiro resultado

    # Exibir a imagem com as detecções de caracteres
    img_with_detections = result.plot()

    # Criar uma pasta para salvar a imagem e o texto
    output_image_folder = os.path.join(output_folder, f"image_{idx}")
    os.makedirs(output_image_folder, exist_ok=True)

    # Salvar a imagem com detecções
    output_image_path = os.path.join(output_image_folder, image_file)
    cv2.imwrite(output_image_path, img_with_detections)

    # Coletar as detecções dos caracteres
    detected_characters = []

    # Loop sobre todas as detecções de caixas delimitadoras
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa
        confidence = box.conf[0].item()  # Confiança da detecção
        class_id = int(box.cls[0].item())  # ID da classe (número ou letra detectada)

        # Obter o caractere correspondente da classe
        character = model.names[class_id]  # O nome da classe será o caractere detectado
        detected_characters.append((x1, character, confidence))  # Armazene coordenada X e caractere

    # Ordenar os caracteres da esquerda para a direita (com base nas coordenadas X)
    detected_characters = sorted(detected_characters, key=lambda x: x[0])

    # Construir a string da placa
    plate_text = ''.join([char for _, char, _ in detected_characters])
    
    # Salvar o texto da placa em um arquivo .txt
    output_text_path = os.path.join(output_image_folder, f"placa_{idx}.txt")
    with open(output_text_path, 'w') as text_file:
        text_file.write(plate_text)

    print(f"Processamento da imagem {image_file} concluído. Texto detectado: {plate_text}")

