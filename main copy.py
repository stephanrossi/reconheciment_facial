import face_recognition
import cv2
import numpy as np
from datetime import datetime
import os
import threading

# Verifica e cria o diretório de logs se necessário
if not os.path.exists('logs'):
    os.makedirs('logs')

# Carregar imagens das faces conhecidas
known_face_encodings = []
known_face_names = []

known_faces_dir = 'C:\\Users\\Casa\\Desktop\\Programação\\python\\know_faces'

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"Não foi possível localizar faces na imagem {filename}")

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
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Converte a imagem de BGR (OpenCV) para RGB (face_recognition)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Localiza todas as faces no frame e seus encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # Compara com as faces conhecidas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"

            # Usa a face conhecida com a menor distância
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # Registra o reconhecimento no log
                    log_recognition(name, camera_name)

            face_names.append(name)

        # Opcional: Exibir o resultado em uma janela
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Redimensiona as coordenadas para o tamanho original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Desenha um retângulo ao redor da face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Escreve o nome abaixo do retângulo
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

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
