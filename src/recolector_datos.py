import cv2
import os
import numpy as np
from mediapipe_utils import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints

# 1. CONFIGURACIÓN MODULAR
print("=== SISTEMA DE RECOLECCIÓN DE LENGUA DE SEÑAS ===")
categoria = input("Ingresa el nombre de la categoría (ej. 01_saludos, 02_numeros): ").strip()
palabra = input("Ingresa la palabra a grabar (ej. hola, gracias): ").strip().lower()

DATA_PATH = os.path.join('..', 'dataset_lsp', categoria, palabra)
no_secuencias = 30      # Cantidad de repeticiones (ejemplos) por palabra
sequence_length = 30    # 30 fotogramas matemáticos (1 segundo) por secuencia

# 2. CREACIÓN DE CARPETAS
for secuencia in range(no_secuencias):
    try:
        os.makedirs(os.path.join(DATA_PATH, f"secuencia_{secuencia}"))
    except OSError:
        pass

# 3. INICIO DE LA CAPTURA EN TIEMPO REAL
cap = cv2.VideoCapture(0) # Inicia tu cámara
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for secuencia in range(no_secuencias):
        memoria_frames = [] # Búfer circular para la secuencia
        
        for num_frame in range(sequence_length):
            ret, frame = cap.read()
            if not ret: continue
            
            # Espejo (flip) para que interactuar sea natural e intuitivo
            frame = cv2.flip(frame, 1)
            
            # Detectar y dibujar landmarks
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # Lógica de protocolo con Pausa Inicial
            if num_frame == 0:
                cv2.putText(image, 'PREPARATE...', (120, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Grabando: {palabra.upper()} - Intento {secuencia+1}/{no_secuencias}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Recoleccion LSP', image)
                cv2.waitKey(2000) # Pausa de 2 segundos
            else:
                cv2.putText(image, f'Grabando: {palabra.upper()} - Intento {secuencia+1}/{no_secuencias}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Recoleccion LSP', image)
            
            # Extracción del vector y guardado temporal
            keypoints = extract_keypoints(results)
            memoria_frames.append(keypoints)
            
            # Para abortar presiona 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        # Guarda el bloque de 30 frames en formato binario .npy
        ruta_guardado = os.path.join(DATA_PATH, f"secuencia_{secuencia}", 'frames_30.npy')
        np.save(ruta_guardado, memoria_frames)
        
    cap.release()
    cv2.destroyAllWindows()

print(f"\n✅ ¡Excelente! Corpus guardado en: {DATA_PATH}")