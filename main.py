import cv2
from fer import FER
import torch
from ultralytics import YOLO
import concurrent.futures

# Inicializar variável para ativar/desativar reconhecimento de emoções
reconhecimento_emocoes = True

# Função de reconhecimento de objetos
def reconhecer_objetos(frame, model, device):
    original_h, original_w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (416, 416)) / 255.0
    frame_tensor = torch.from_numpy(frame_resized).float().to(device)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        results = model(frame_tensor)
    
    detections = results[0].boxes
    objetos_detectados = []

    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0].tolist()
        score = det.conf[0].item()
        class_id = int(det.cls[0].item())

        if score > 0.6:
            nome_objeto = model.names[class_id]

            # Ajustar coordenadas para manter a proporção correta
            x1, y1, x2, y2 = int(x1 * original_w / 416), int(y1 * original_h / 416), \
                             int(x2 * original_w / 416), int(y2 * original_h / 416)

            objetos_detectados.append((nome_objeto, (x1, y1, x2, y2), score))

    return objetos_detectados

# Função de reconhecimento de emoções
def detectar_emocoes(frame, faces, detector):
    emoções_detectadas = []
    for (x, y, w, h) in faces:
        rosto = frame[y:y+h, x:x+w]
        emoções = detector.top_emotion(rosto)
        if emoções and emoções[0]:
            emocion = emoções[0][0]
            emoções_detectadas.append((x, y, w, h, emocion))  # Guardar coordenadas do rosto + emoção
    return emoções_detectadas

# Inicializar modelos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov5n.pt').to(device)
detector = FER()

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Função para desenhar texto com melhor aparência
def desenhar_texto_profissional(frame, texto, pos, cor_fundo, cor_texto, fonte=cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte=0.7, espessura=2):
    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, tamanho_fonte, espessura)
    cv2.rectangle(frame, (pos[0] - 5, pos[1] - 10), (pos[0] + largura_texto + 5, pos[1] + altura_texto + 10), cor_fundo, -1)
    cv2.putText(frame, texto, (pos[0], pos[1] + altura_texto), fonte, tamanho_fonte, cor_texto, espessura)

# Loop de captura de vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        objetos_future = executor.submit(reconhecer_objetos, frame, model, device)
        objetos_resultado = objetos_future.result()

        if reconhecimento_emocoes:  # Verifica se o reconhecimento de emoções está ativado
            emocoes_future = executor.submit(detectar_emocoes, frame, faces, detector)
            emocoes_resultado = emocoes_future.result()
        else:
            emocoes_resultado = []

    # Exibir objetos detectados
    for objeto_nome, (x1, y1, x2, y2), score in objetos_resultado:
        texto_objeto = f"{objeto_nome} ({score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caixa verde ao redor do objeto
        desenhar_texto_profissional(frame, texto_objeto, (x1, max(20, y1 - 15)), (0, 255, 0), (0, 0, 0))

    # Exibir emoções detectadas
    for (x, y, w, h, emocion) in emocoes_resultado:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Caixa azul ao redor do rosto
        desenhar_texto_profissional(frame, f"Emotion: {emocion}", (x, y - 10), (255, 255, 0), (0, 0, 0))

    # Mostrar o frame com detecções
    cv2.imshow('Object and Emotion Recognition', frame)

    # Comandos do teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # Pressione 'q' para sair
    elif key == ord('e'):
        reconhecimento_emocoes = not reconhecimento_emocoes  # Alternar reconhecimento de emoções

cap.release()
cv2.destroyAllWindows()
