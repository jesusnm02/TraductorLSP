import cv2
import numpy as np
import os
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
from mediapipe_utils import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

mp_drawing = mp.solutions.drawing_utils
app = Flask(__name__)

# ==========================================
# 1. VARIABLES GLOBALES DEL SISTEMA
# ==========================================
categoria_actual = ""
tipo_modelo_actual = "dinamico"
model = None
acciones = []
secuencia = []
traduccion = ""
umbral = 0.85

# Variables de Grabación y Entrenamiento
modo_grabacion = False
estado_rec = {"categoria": "", "palabra": "", "sec": 0, "fase": "reposo", "tiempo_ini": 0}
frames_grabacion = []
estado_entrenamiento = {"entrenando": False, "progreso": 0, "mensaje": "Esperando..."}
capturar_fotograma_actual = False

# NUEVO: Variables para el Filtro de Estabilidad y Delay Asíncrono
tiempo_ultima_traduccion = 0.0
buffer_estatico = []

# ==========================================
# 2. CARGA DINÁMICA DE MODELOS
# ==========================================
def cargar_modelo(cat):
    global model, acciones, tipo_modelo_actual
    if not cat: return False
    
    tipo_modelo_actual = "estatico" if "estatico" in cat else "dinamico"
    ruta_datos = os.path.join('..', 'dataset_lsp', cat)
    ruta_modelo = os.path.join('..', 'models', f'modelo_{cat}.h5')
    
    if os.path.exists(ruta_datos):
        acciones = np.array(os.listdir(ruta_datos))
    else:
        acciones = []
        
    try:
        model = load_model(ruta_modelo)
        print(f"✅ Modelo '{cat}' ({tipo_modelo_actual}) cargado exitosamente.")
        return True
    except:
        model = None
        print(f"⚠️ No hay modelo entrenado para '{cat}'.")
        return False

class ProgresoIA(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global estado_entrenamiento
        estado_entrenamiento["progreso"] = epoch + 1

def extraer_solo_manos(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# ==========================================
# 3. MOTOR DE VISIÓN Y CÁMARA (TIEMPO REAL)
# ==========================================
def generar_frames():
    global secuencia, traduccion, model, acciones, modo_grabacion, estado_rec, frames_grabacion, tipo_modelo_actual, capturar_fotograma_actual
    global tiempo_ultima_traduccion, buffer_estatico
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            
            es_estatico_grabando = modo_grabacion and "estatico" in estado_rec["categoria"]
            es_estatico_infiriendo = not modo_grabacion and tipo_modelo_actual == "estatico"
            
            if es_estatico_grabando or es_estatico_infiriendo:
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            else:
                draw_styled_landmarks(image, results)
            
            # --- MODO RECOLECCIÓN ---
            if modo_grabacion:
                tiempo_actual = time.time()
                if estado_rec["fase"] == "preparacion":
                    if es_estatico_grabando:
                        estado_rec["fase"] = "grabando"
                        frames_grabacion = []
                    else:
                        cv2.putText(image, f"PREPARATE: {estado_rec['palabra']} ({estado_rec['sec']+1}/30)", (15, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        if tiempo_actual - estado_rec["tiempo_ini"] > 2.0:
                            estado_rec["fase"] = "grabando"
                            frames_grabacion = []
                        
                elif estado_rec["fase"] == "grabando":
                    if es_estatico_grabando:
                        cv2.putText(image, f"MODO FOTO: {estado_rec['palabra']} ({estado_rec['sec']}/30)", (15, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        if capturar_fotograma_actual:
                            frames_grabacion.append(extraer_solo_manos(results))
                            estado_rec["sec"] += 1
                            capturar_fotograma_actual = False 
                            
                        if estado_rec["sec"] >= 30:
                            ruta_g = os.path.join('..', 'dataset_lsp', estado_rec["categoria"], estado_rec["palabra"])
                            os.makedirs(ruta_g, exist_ok=True)
                            np.save(os.path.join(ruta_g, f"muestras_{int(time.time())}.npy"), np.array(frames_grabacion))
                            modo_grabacion = False
                    else:
                        cv2.putText(image, f"GRABANDO VIDEO: {estado_rec['palabra']}...", (15, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        frames_grabacion.append(extract_keypoints(results))
                        if len(frames_grabacion) == 30:
                            ruta_g = os.path.join('..', 'dataset_lsp', estado_rec["categoria"], estado_rec["palabra"], f"sec_{estado_rec['sec']}_{int(time.time())}")
                            os.makedirs(ruta_g, exist_ok=True)
                            np.save(os.path.join(ruta_g, "frames_30.npy"), np.array(frames_grabacion))
                            estado_rec["sec"] += 1
                            frames_grabacion = []
                            if estado_rec["sec"] >= 30:
                                modo_grabacion = False
                            else:
                                estado_rec["fase"] = "preparacion"
                                estado_rec["tiempo_ini"] = time.time()
                                
            # --- MODO TRADUCCIÓN ---
            else:
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    if tipo_modelo_actual == "estatico":
                        tiempo_actual = time.time()
                        
                        # 1. DELAY ASÍNCRONO: Evaluar solo si pasaron 2 segundos desde la última predicción
                        if tiempo_actual - tiempo_ultima_traduccion > 2.0:
                            keypoints = extraer_solo_manos(results)
                            if model is not None and len(acciones) > 0:
                                res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)
                                if np.max(res) > umbral:
                                    palabra_detectada = acciones[np.argmax(res)]
                                    
                                    # 2. FILTRO DE ESTABILIDAD: Exigir 5 fotogramas idénticos (ignora transiciones)
                                    buffer_estatico.append(palabra_detectada)
                                    buffer_estatico = buffer_estatico[-5:]
                                    
                                    if len(buffer_estatico) == 5 and all(p == palabra_detectada for p in buffer_estatico):
                                        if traduccion != palabra_detectada:
                                            traduccion = palabra_detectada
                                            # Activar el delay de 2 segundos sin congelar la cámara
                                            tiempo_ultima_traduccion = tiempo_actual 
                                            buffer_estatico = [] # Limpiar el buffer
                        else:
                            # Vaciar el buffer mientras dura el delay para no acumular ruido
                            buffer_estatico = []
                    else:
                        keypoints = extract_keypoints(results)
                        secuencia.append(keypoints)
                        secuencia = secuencia[-30:]
                        if len(secuencia) == 30 and model is not None and len(acciones) > 0:
                            res = model.predict(np.expand_dims(secuencia, axis=0), verbose=0)
                            if np.max(res) > umbral:
                                palabra = acciones[np.argmax(res)]
                                if traduccion != palabra:
                                    traduccion = palabra
                else:
                    secuencia = []
                    buffer_estatico = []

            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 4. RUTAS WEB Y API
# ==========================================
@app.route('/')
def index():
    if not os.path.exists(os.path.join('..', 'dataset_lsp')): os.makedirs(os.path.join('..', 'dataset_lsp'))
    categorias_disp = os.listdir(os.path.join('..', 'dataset_lsp'))
    return render_template('index.html', categorias=categorias_disp)

@app.route('/video_feed')
def video_feed(): return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/traduccion_actual')
def traduccion_actual_api(): return jsonify({'palabra': traduccion if not modo_grabacion else "GRABANDO..."})

@app.route('/resetear_traduccion', methods=['POST'])
def resetear_traduccion():
    global traduccion, secuencia, buffer_estatico
    traduccion = "" 
    secuencia = []
    buffer_estatico = []
    return jsonify({'status': 'ok'})

@app.route('/cambiar_categoria', methods=['POST'])
def cambiar_categoria():
    global categoria_actual, traduccion, secuencia, buffer_estatico, tiempo_ultima_traduccion
    data = request.get_json()
    nueva_cat = data['categoria']
    if cargar_modelo(nueva_cat):
        categoria_actual = nueva_cat
        traduccion = "" 
        secuencia = []
        buffer_estatico = []
        tiempo_ultima_traduccion = 0.0
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error', 'mensaje': f'El modelo {nueva_cat} no existe o no ha sido entrenado.'})

@app.route('/iniciar_grabacion', methods=['POST'])
def iniciar_grabacion():
    global modo_grabacion, estado_rec
    data = request.get_json()
    estado_rec.update({"categoria": data['categoria'], "palabra": data['palabra'], "sec": 0, "fase": "preparacion", "tiempo_ini": time.time()})
    modo_grabacion = True
    return jsonify({'status': 'ok'})

@app.route('/estado_grabacion')
def obtener_estado_grabacion():
    return jsonify({'grabando': modo_grabacion, 'progreso': estado_rec["sec"]})

@app.route('/capturar_foto', methods=['POST'])
def capturar_foto():
    global capturar_fotograma_actual
    capturar_fotograma_actual = True
    return jsonify({'status': 'ok'})

@app.route('/progreso_entrenamiento')
def obtener_progreso_entrenamiento():
    return jsonify(estado_entrenamiento)

# ==========================================
# 5. MOTOR DE ENTRENAMIENTO IA (HÍBRIDO)
# ==========================================
def hilo_entrenamiento_ia(cat):
    global estado_entrenamiento
    estado_entrenamiento.update({"entrenando": True, "progreso": 0, "mensaje": "Extrayendo tensores matemáticos..."})
    
    ruta_datos = os.path.join('..', 'dataset_lsp', cat)
    labels = np.array(os.listdir(ruta_datos))
    secuencias, etiquetas = [], []
    mapa_etiquetas = {etiqueta:num for num, etiqueta in enumerate(labels)}
    
    es_estatico = "estatico" in cat
    
    for accion in labels:
        ruta_accion = os.path.join(ruta_datos, accion)
        if es_estatico:
            for arch in os.listdir(ruta_accion):
                if arch.endswith('.npy'):
                    res = np.load(os.path.join(ruta_accion, arch))
                    for frame in res:
                        secuencias.append(frame)
                        etiquetas.append(mapa_etiquetas[accion])
        else:
            for seq_folder in os.listdir(ruta_accion):
                ruta_npy = os.path.join(ruta_accion, seq_folder, "frames_30.npy")
                if os.path.exists(ruta_npy):
                    res = np.load(ruta_npy)
                    secuencias.append(res)
                    etiquetas.append(mapa_etiquetas[accion])
                
    if len(secuencias) == 0:
         estado_entrenamiento.update({"entrenando": False, "mensaje": "Error: Datos insuficientes."})
         return

    X = np.array(secuencias)
    y = to_categorical(etiquetas).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    estado_entrenamiento["mensaje"] = f"Construyendo Cerebro ({'MLP' if es_estatico else 'LSTM'})..."
    nuevo_modelo = Sequential()
    
    if es_estatico:
        nuevo_modelo.add(Dense(256, activation='relu', input_shape=(126,)))
        nuevo_modelo.add(Dropout(0.3))
        nuevo_modelo.add(Dense(128, activation='relu'))
        nuevo_modelo.add(Dropout(0.3))
        nuevo_modelo.add(Dense(len(labels), activation='softmax'))
    else:
        nuevo_modelo.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 258)))
        nuevo_modelo.add(LSTM(128, return_sequences=False, activation='tanh'))
        nuevo_modelo.add(Dense(64, activation='relu'))
        nuevo_modelo.add(Dense(len(labels), activation='softmax'))
    
    nuevo_modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    estado_entrenamiento["mensaje"] = f"Entrenando {len(labels)} palabras de '{cat}'..."
    nuevo_modelo.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0, callbacks=[ProgresoIA()])
    
    estado_entrenamiento["mensaje"] = "Guardando modelo neuronal..."
    os.makedirs(os.path.join('..', 'models'), exist_ok=True)
    nuevo_modelo.save(os.path.join('..', 'models', f'modelo_{cat}.h5'))
    
    cargar_modelo(cat) 
    estado_entrenamiento.update({"entrenando": False, "mensaje": "✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!"})

@app.route('/entrenar_modelo', methods=['POST'])
def entrenar():
    data = request.get_json()
    threading.Thread(target=hilo_entrenamiento_ia, args=(data['categoria'],)).start()
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5050)