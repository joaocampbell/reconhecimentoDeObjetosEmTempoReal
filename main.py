import cv2
from fer import FER
import torch
from ultralytics import YOLO
import concurrent.futures

# Função de reconhecimento de objetos (sem Watson)
def reconhecer_objetos(frame, model, device):
    original_h, original_w = frame.shape[:2]  # Obter as dimensões originais da imagem
    frame_resized = cv2.resize(frame, (416, 416))  # Resolução reduzida para melhorar desempenho
    frame_resized = frame_resized / 255.0  # Normalizar a imagem
    frame_tensor = torch.from_numpy(frame_resized).float().to(device)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # Transformar em BCHW

    # Enviar para o modelo YOLO
    results = model(frame_tensor)  # Obtenha os resultados diretamente
    detections = results[0].boxes  # Obter as caixas de detecção

    # Filtro para detecções
    objetos_detectados = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0].tolist()  # Coordenadas da caixa
        score = det.conf[0].item()  # Pontuação de confiança
        class_id = int(det.cls[0].item())  # ID da classe

        if score > 0.6:  # Apenas objetos com confiança acima de 0.6
            nome_objeto = model.names[class_id]  # Nome do objeto

            # Ajustar as coordenadas da caixa para o tamanho original da imagem
            x1, y1, x2, y2 = int(x1 * original_w / 416), int(y1 * original_h / 416), \
                             int(x2 * original_w / 416), int(y2 * original_h / 416)

            objetos_detectados.append((nome_objeto, (x1, y1, x2, y2), score))

    return objetos_detectados

# Função de reconhecimento de emoções
def detectar_emocoes(frame, faces, detector):
    emoções_detectadas = []
    for (x, y, w, h) in faces:
        rosto = frame[y:y+h, x:x+w]  # Captura a região do rosto
        # Usar o detector FER para identificar a emoção
        emoções = detector.top_emotion(rosto)
        if emoções and emoções[0]:
            emocion = emoções[0][0]  # Emoção detectada
            emoções_detectadas.append((x, y, emocion))
    return emoções_detectadas

# Inicializar o modelo YOLO (usando versão leve)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov5n.pt').to(device)  # Usando YOLOv5n para melhor desempenho

# Inicializar o modelo FER para reconhecimento de emoções
detector = FER()

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)  # Ajuste de FPS para desempenho
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Resolução ajustada
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Função para desenhar o texto de forma profissional
def desenhar_texto_profissional(frame, texto, pos, cor_fundo, cor_texto, fonte=cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte=0.8, espessura=2):
    # Dimensões do texto
    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, tamanho_fonte, espessura)
    
    # Criar fundo do texto com borda arredondada
    cv2.rectangle(frame, (pos[0] - 5, pos[1] - 10), (pos[0] + largura_texto + 5, pos[1] + altura_texto + 10), cor_fundo, -1)
    
    # Colocar o texto sobre o fundo
    cv2.putText(frame, texto, (pos[0], pos[1] + altura_texto), fonte, tamanho_fonte, cor_texto, espessura)

# Loop de captura de vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Se não conseguir capturar, sair do loop

    # Detecção de rostos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Processar objetos e emoções de forma paralela
    with concurrent.futures.ThreadPoolExecutor() as executor:
        emocoes_future = executor.submit(detectar_emocoes, frame, faces, detector)
        objetos_future = executor.submit(reconhecer_objetos, frame, model, device)

        objetos_resultado = objetos_future.result()
        emocoes_resultado = emocoes_future.result()

    # Exibir as emoções detectadas de forma profissional
    for (x, y, emocion) in emocoes_resultado:
        # Desenhando o texto de emoção de forma profissional
        texto_emocao = f"Emotion: {emocion}"
        desenhar_texto_profissional(frame, texto_emocao, (x, y - 10), (255, 255, 0), (0, 0, 0))

    # Exibir os objetos detectados de forma mais profissional
    for objeto_nome, (x1, y1, x2, y2), score in objetos_resultado:
        texto_objeto = f"{objeto_nome} ({score:.2f})"
        desenhar_texto_profissional(frame, texto_objeto, (x1, y1 - 10), (0, 255, 0), (0, 0, 0))

    # Mostrar a imagem com as detecções e emoções
    cv2.imshow('Object and Emotion Recognition', frame)

    # Fechar com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Pressione 'q' para sair

cap.release()
cv2.destroyAllWindows()
