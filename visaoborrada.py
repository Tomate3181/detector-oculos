import cv2
import numpy as np
import time
import os

# --- CONFIGURAÇÕES DO MODELO DE IA (DEEP LEARNING) ---
# Certifique-se de que estes arquivos estão na mesma pasta do script!
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

# Verifica se os arquivos existem para evitar erro obscuro
if not os.path.exists(PROTO_PATH) or not os.path.exists(MODEL_PATH):
    print("ERRO: Arquivos do modelo DNN não encontrados!")
    print(f"Certifique-se de baixar '{PROTO_PATH}' e '{MODEL_PATH}' e colocá-los na mesma pasta.")
    exit()

# Carrega a rede neural
print("[INFO] Carregando modelo de detecção de rostos ultra robusto...")
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

# Se tiver GPU NVIDIA configurada com OpenCV, descomente as linhas abaixo para FPS máximo
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def apply_pixelate_filter(img, x, y, w, h):
    """Aplica efeito pixelado (estilo AR) em uma região."""
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        # Pixelate: diminui e aumenta com vizinho mais próximo
        temp = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
        img[y:y+h, x:x+w] = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return img

def main():
    # Inicializa a câmera (use 0, 1 ou a URL do Iriun/DroidCam)
    # cap = cv2.VideoCapture(0) # Webcam local
    cap = cv2.VideoCapture(0) # Tente mudar para 1 ou 2 se usar celular via USB

    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a câmera.")
        exit()

    prev_time = 0
    print("[INFO] Sistema de Privacidade Ativo. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        
        # --- PREPARAÇÃO DA IMAGEM PARA A IA ---
        # A rede precisa de uma imagem de 300x300 pixels, com subtração de média
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))

        # --- DETECÇÃO DE ROSTOS (DEEP LEARNING) ---
        net.setInput(blob)
        detections = net.forward()

        # Itera sobre as detecções
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filtra detecções fracas (Limiar de confiança de 50%)
            if confidence > 0.5:
                # Computa as coordenadas do box delimitador
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Garante que as coordenadas estão dentro do frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # Aplica o filtro de privacidade (Pixelate)
                frame = apply_pixelate_filter(frame, startX, startY, endX - startX, endY - startY)

                # Desenha o box e a confiança (útil para o Pitch mostrar a IA funcionando)
                text = f"PESSOA: {confidence * 100:.1f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # --- CÁLCULO DE FPS E EXIBIÇÃO ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AR Privacy Shield (Deep Learning DNN)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()