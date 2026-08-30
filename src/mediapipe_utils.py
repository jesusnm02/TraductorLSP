import cv2
import mediapipe as mp
import numpy as np

# Inicializamos el modelo Holistic de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Convierte el espacio de color y procesa la imagen con MediaPipe"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # MediaPipe requiere RGB
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Volvemos a BGR para OpenCV
    return image, results

def draw_styled_landmarks(image, results):
    """Dibuja las conexiones en la pantalla para dar retroalimentación visual al usuario"""
    # Dibuja la postura (cuerpo)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Dibuja la mano izquierda
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Dibuja la mano derecha
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    """Extrae y concatena las coordenadas (x,y,z) en un solo vector numérico"""
    # Si detecta la postura, extrae; si no, rellena con ceros (33 puntos * 4 valores = 132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Mano Izquierda (21 puntos * 3 = 63)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Mano Derecha (21 puntos * 3 = 63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])