import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

import cv2
import numpy as np
import os
from mediapipe_utils import mp_holistic, mediapipe_detection, draw_styled_landmarks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("=== SISTEMA DE TRADUCCIÓN ESTÁTICA (SOLO MANOS) ===")
categoria = "numeros" # Tu categoría estática
ruta_datos = os.path.join('..', 'dataset_lsp', categoria)
acciones = np.array(os.listdir(ruta_datos))
mapa_etiquetas = {etiqueta:num for num, etiqueta in enumerate(acciones)}

# ==========================================
# 1. EXTRACCIÓN DE DATA EXISTENTE (FILTRANDO SOLO MANOS)
# ==========================================
secuencias, etiquetas = [], []
print("Extrayendo características... (Ignorando cuerpo, conservando solo manos)")

for accion in acciones:
    for seq in range(30): # Las 30 carpetas que ya grabaste
        try:
            # Cargamos la matriz original de (30, 258)
            res = np.load(os.path.join(ruta_datos, accion, f"secuencia_{seq}", "frames_30.npy"))
            # MAGIA MATEMÁTICA: Tomamos solo el primer fotograma (0) 
            # y cortamos desde la posición 132 hasta la 258 (los 126 puntos exactos de las manos)
            manos_estaticas = res[0, 132:] 
            secuencias.append(manos_estaticas)
            etiquetas.append(mapa_etiquetas[accion])
        except:
            pass

X = np.array(secuencias) # Shape será (muestras, 126)
y = to_categorical(etiquetas).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ==========================================
# 2. NUEVO CEREBRO (PERCEPTRÓN MULTICAPA - MLP)
# ==========================================
print("\nConstruyendo Red Neuronal Densa (MLP) para señas estáticas...")
model = Sequential()
# Capa de entrada de exactamente 126 características (solo manos)
model.add(Dense(256, activation='relu', input_shape=(126,)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(acciones), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Entrenando... (Será casi instantáneo)")
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

# ==========================================
# 3. TRADUCCIÓN EN TIEMPO REAL
# ==========================================
def extraer_solo_manos(results):
    """ Función optimizada que ignora el cuerpo y extrae solo 126 puntos de las manos """
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh]) # Exactamente 126 puntos

print("\n✅ Entrenado. Encendiendo cámara para traducción en vivo...")
cap = cv2.VideoCapture(0)
traduccion = ""

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        
        # Dibujamos SOLO las manos en la pantalla para no saturar tu vista
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Si detecta al menos una mano en la pantalla
        if results.left_hand_landmarks or results.right_hand_landmarks:
            # Extrae 1 solo fotograma de 126 puntos
            keypoints = extraer_solo_manos(results) 
            
            # Predicción instantánea (sin buffers de 30 frames)
            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)
            
            if np.max(res) > 0.85: # 85% de confianza
                traduccion = acciones[np.argmax(res)]
        else:
            traduccion = "..."

        # Mostrar en pantalla
        cv2.rectangle(image, (0,0), (640, 60), (245, 117, 16), -1)
        cv2.putText(image, f"Numero: {traduccion.upper()}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        cv2.imshow('Traductor Estatico - Numeros', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()