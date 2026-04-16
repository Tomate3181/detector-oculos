import cv2
import numpy as np
import time

# --- CONFIGURAÇÕES ---
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5

# Carrega a rede neural
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
qr_detector = cv2.QRCodeDetector()

def draw_ar_ui(img, x, y, w, h, label, color):
    """Desenha uma interface de óculos AR ao redor do objeto."""
    t = 2 # espessura
    l = 20 # comprimento do canto
    
    # Cantos da moldura (Estilo Mira AR)
    cv2.line(img, (x, y), (x + l, y), color, t)
    cv2.line(img, (x, y), (x, y + l), color, t)
    
    cv2.line(img, (x + w, y), (x + w - l, y), color, t)
    cv2.line(img, (x + w, y), (x + w, y + l), color, t)
    
    cv2.line(img, (x, y + h), (x + l, y + h), color, t)
    cv2.line(img, (x, y + h), (x, y + h - l), color, t)
    
    cv2.line(img, (x + w, y + h), (x + w - l, y + h), color, t)
    cv2.line(img, (x + w, y + h), (x + w, y + h - l), color, t)
    
    # Label com fundo semi-transparente
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(img, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
    cv2.putText(img, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def apply_strong_blur(img, x, y, w, h):
    """Aplica um borrão denso com travas de segurança contra crash."""
    # 1. Garante que as coordenadas estão dentro dos limites da imagem
    img_h, img_w = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    # 2. Verifica se a região tem um tamanho válido (maior que zero)
    if w > 1 and h > 1:
        roi = img[y:y+h, x:x+w]
        
        # 3. Verifica se a ROI não está vazia de fato
        if roi is not None and roi.size > 0:
            try:
                # O kernel do stackBlur precisa ser ímpar e > 0
                blur_size = 91
                blur = cv2.stackBlur(roi, (blur_size, blur_size))
                img[y:y+h, x:x+w] = blur
            except cv2.error as e:
                print(f"[AVISO] Falha ao borrar região: {e}")
    
    return img

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Efeito de "Scan" global (opcional: escurece um pouco a imagem para o HUD brilhar)
        # frame = cv2.addWeighted(frame, 0.8, np.zeros(frame.shape, frame.dtype), 0, 0)
        
        h, w = frame.shape[:2]
        display_frame = frame.copy()

        # 1. ROSTOS (DNN)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                
                apply_strong_blur(display_frame, x1, y1, x2-x1, y2-y1)
                draw_ar_ui(display_frame, x1, y1, x2-x1, y2-y1, f"PESSOA {int(confidence*100)}%", (0, 255, 255))

        # 2. QR CODES
        ok, points = qr_detector.detect(frame)
        if ok and points is not None:
            pts = points[0].astype(int)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            apply_strong_blur(display_frame, x_min, y_min, x_max-x_min, y_max-y_min)
            draw_ar_ui(display_frame, x_min, y_min, x_max-x_min, y_max-y_min, "DADO SENSIVEL (QR)", (0, 0, 255))

        # 3. TELAS E DOCUMENTOS (Processamento Avançado)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 30, 150)
        
        # Dilatação para fechar os contornos de telas/celulares
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        edged = cv2.dilate(edged, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10000 < area < (w * h * 0.8): # Nem muito pequeno, nem o frame todo
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True) # 0.04 aumenta a tolerância
                
                if len(approx) == 4: # Retângulos
                    x, y, bw, bh = cv2.boundingRect(approx)
                    aspect_ratio = bw / float(bh)
                    # Filtra proporções comuns de telas (celular em pé ou deitado / monitor)
                    if 0.4 < aspect_ratio < 2.5:
                        apply_strong_blur(display_frame, x, y, bw, bh)
                        draw_ar_ui(display_frame, x, y, bw, bh, "CONTEUDO RESTRITO", (0, 255, 0))

        # UI do Sistema (Cantos da tela)
        cv2.putText(display_frame, "MODO PRIVACIDADE: ATIVO", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Cálculo de FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(display_frame, f"SYS_LATENCY: {int((1/fps)*1000)}ms", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("AR Privacy OS v1.0", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()