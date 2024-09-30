import dlib
import cv2
import numpy as np
import os
from datetime import datetime
import threading

# Verifica e cria o diretório de logs se necessário
if not os.path.exists('logs'):
    os.makedirs('logs')

# Verifica e cria o diretório de modelos se necessário
if not os.path.exists('models'):
    os.makedirs('models')

# Verifica e cria o diretório para rostos desconhecidos se necessário
if not os.path.exists('unknown_faces'):
    os.makedirs('unknown_faces')

# Caminhos para os modelos pré-treinados
PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"

# Verifica se os modelos existem
if not os.path.exists(PREDICTOR_PATH) or not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
    print("Por favor, certifique-se de que os modelos pré-treinados estão no diretório 'models'.")
    exit()

# Carregar os modelos
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Diretório das faces conhecidas
known_faces_dir = 'C:\\Users\\Casa\\Desktop\\Programacao\\python\\know_faces'

# Listas para armazenar os descritores e nomes das faces conhecidas
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(known_faces_dir, filename)
        image = cv2.imread(image_path)

        # Verifica se a imagem foi carregada corretamente
        if image is None:
            print(f"Não foi possível carregar a imagem {filename}. Verifique se o arquivo existe e está em um formato suportado.")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = detector(rgb_image, 1)  # Upsampling = 1
        if len(detections) == 0:
            print(f"Não foi possível localizar faces na imagem {filename}")
            continue

        # Assume que há apenas uma face por imagem de referência
        shape = shape_predictor(rgb_image, detections[0])
        face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
        known_face_encodings.append(np.array(face_descriptor))
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)

# Lista de URLs RTSP das câmeras
rtsp_urls = {
    'Camera 1': 'rtsp://admin:a3e1lm2s2y@192.168.15.103:10554/tcp/av0_0',
    # 'Camera 2': 'rtsp://usuario:senha@ip_camera2/stream1',
    # Adicione mais câmeras conforme necessário
}

def log_recognition(name, camera_name):
    now = datetime.now()
    timestamp = now.strftime('%d/%m/%Y %H:%M:%S')
    log_message = f"{timestamp} - {name} reconhecido na {camera_name}\n"
    print(log_message.strip())

    log_file = 'logs/reconhecimentos.txt'
    with open(log_file, 'a') as file:
        file.write(log_message)

def save_unknown_face(face_image, camera_name):
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{camera_name}.jpg"
    filepath = os.path.join('unknown_faces', filename)
    cv2.imwrite(filepath, face_image)
    print(f"Rosto desconhecido salvo em {filepath}")

def process_camera(camera_name, rtsp_url):
    video_capture = cv2.VideoCapture(rtsp_url)

    if not video_capture.isOpened():
        print(f"Não foi possível conectar à {camera_name}")
        return

    print(f"Processando {camera_name}...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Falha ao ler frame da {camera_name}")
            break

        # Redimensiona o frame para acelerar o processamento
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detecção de faces com upsampling = 1
        detections = detector(rgb_small_frame, 1)

        face_names = []
        face_locations = []

        # Desenhar retângulo azul e exibir 'Processando...' para cada face detectada
        for detection in detections:
            # Escala as coordenadas para o tamanho original
            left = detection.left() * 2
            top = detection.top() * 2
            right = detection.right() * 2
            bottom = detection.bottom() * 2

            # Desenha o retângulo azul
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)  # Azul
            # Escreve 'Processando...'
            cv2.putText(frame, 'Processando...', (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)

        # Exibe o frame com 'Processando...'
        cv2.imshow(camera_name, frame)
        cv2.waitKey(1)  # Pequeno delay para atualizar a janela

        # Processamento e reconhecimento
        for detection in detections:
            # Pontos de referência faciais
            shape = shape_predictor(rgb_small_frame, detection)
            # Descritor facial
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_small_frame, shape)
            face_descriptor_np = np.array(face_descriptor)

            # Comparação com as faces conhecidas
            distances = np.linalg.norm(known_face_encodings - face_descriptor_np, axis=1)
            min_distance = np.min(distances)
            min_distance_index = np.argmin(distances)

            # Definir um limiar de reconhecimento (ajuste conforme necessário)
            threshold = 0.6

            if min_distance < threshold:
                name = known_face_names[min_distance_index]
                # Registra o reconhecimento no log
                log_recognition(name, camera_name)
            else:
                name = "Desconhecido"
                # Salva o rosto desconhecido
                # Escala as coordenadas para o tamanho original
                left = detection.left() * 2
                top = detection.top() * 2
                right = detection.right() * 2
                bottom = detection.bottom() * 2

                # Garante que as coordenadas estão dentro dos limites da imagem
                h, w, _ = frame.shape
                left = max(0, left)
                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)

                # Extrai o rosto da imagem original
                face_image = frame[top:bottom, left:right]
                # Salva o rosto desconhecido
                save_unknown_face(face_image, camera_name)

            face_names.append((detection, name))

        # Atualizar o frame com os resultados
        for detection, name in face_names:
            left = detection.left() * 2
            top = detection.top() * 2
            right = detection.right() * 2
            bottom = detection.bottom() * 2

            # Escolhe a cor do retângulo
            if name != "Desconhecido":
                color = (0, 255, 0)  # Verde
            else:
                color = (0, 0, 255)  # Vermelho

            # Desenha o retângulo
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # Escreve o nome
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Exibe o frame atualizado
        cv2.imshow(camera_name, frame)

        # Sai do loop ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Processa todas as câmeras simultaneamente usando threading
threads = []

for camera_name, rtsp_url in rtsp_urls.items():
    t = threading.Thread(target=process_camera, args=(camera_name, rtsp_url))
    t.start()
    threads.append(t)

# Opcional: Espera todas as threads terminarem
for t in threads:
    t.join()
