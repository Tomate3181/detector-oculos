import cv2
import numpy as np
import time

# --- CONFIGURAÇÕES ---
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

def apply_strong_blur(img, x, y, w, h):
    """Borrão seguro com travas de limite."""
    ih, iw = img.shape[:2]
    x, y, w, h = max(0, x), max(0, y), min(w, iw-x), min(h, ih-y)
    if w > 5 and h > 5:
        roi = img[y:y+h, x:x+w]
        if roi.size > 0:
            img[y:y+h, x:x+w] = cv2.stackBlur(roi, (91, 91))
    return img

def is_valid_screen(approx, bw, bh):
    """Valida se o retângulo detectado tem proporções de tela/documento."""
    if bh == 0: return False
    ar = bw / float(bh)
    
    # Proporções aceitáveis: 0.5 (celular em pé) até 2.2 (monitor ultra-wide)
    if not (0.4 < ar < 2.5):
        return False
        
    # Verifica a "convexidade" - telas são retângulos perfeitos, não formas estranhas
    area = cv2.contourArea(approx)
    hull = cv2.convexHull(approx)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False
    
    solidity = float(area) / hull_area
    if solidity < 0.9: # Se for menor que 90%, a forma é muito irregular para ser uma tela
        return False
        
    return True

def main():
    cap = cv2.VideoCapture(0) # Mude para 1 ou URL se necessário
    
    # Histórico para estabilidade (Dicionário para rastrear detecções)
    screen_history = [] 

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: continue
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # --- DETECÇÃO DE ROSTOS (JÁ ESTÁ PRECISA) ---
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] > 0.6: # Aumentei o threshold para 60%
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                apply_strong_blur(display_frame, x1, y1, x2-x1, y2-y1)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # --- DETECÇÃO DE TELAS COM FILTROS RÍGIDOS ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur mais forte antes do Canny ajuda a ignorar detalhes irrelevantes dentro da tela
        blurred = cv2.medianBlur(gray, 7) 
        edged = cv2.Canny(blurred, 50, 200)
        
        # Operação morfológica para unir bordas quebradas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_screens = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 15000: # Exige um tamanho mínimo considerável
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                if len(approx) == 4:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    if is_valid_screen(approx, bw, bh):
                        # Se passou em todos os filtros, adicionamos à lista
                        current_screens.append((x, y, bw, bh))
                        apply_strong_blur(display_frame, x, y, bw, bh)
                        cv2.drawContours(display_frame, [approx], -1, (0, 255, 0), 2)
                        cv2.putText(display_frame, "SCREEN DETECTED", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Privacy Shield AR - High Precision", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

main()