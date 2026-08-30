import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("=== ENTRENADOR INTELIGENTE DE IA ===")
# 1. CONFIGURACIÓN DINÁMICA
categoria = input("Ingresa el nombre de la categoría a entrenar (ej. saludos, numeros): ").strip()
ruta_datos = os.path.join('..', 'dataset_lsp', categoria)

if not os.path.exists(ruta_datos):
    print(f"❌ Error: No se encontró la carpeta para la categoría '{categoria}'.")
    exit()

acciones = np.array(os.listdir(ruta_datos))
no_secuencias = 30
secuencia_longitud = 30

print(f"=== INICIANDO ENTRENAMIENTO PARA '{categoria.upper()}' ===")
print(f"Palabras detectadas: {acciones}")

# 2. CARGAR Y ETIQUETAR LOS DATOS
mapa_etiquetas = {etiqueta:num for num, etiqueta in enumerate(acciones)}
secuencias, etiquetas = [], []

for accion in acciones:
    for secuencia in range(no_secuencias):
        try:
            res = np.load(os.path.join(ruta_datos, accion, f"secuencia_{secuencia}", "frames_30.npy"))
            secuencias.append(res)
            etiquetas.append(mapa_etiquetas[accion])
        except:
            pass # Ignora si falta algún frame

X = np.array(secuencias)
y = to_categorical(etiquetas).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. CONSTRUCCIÓN DE LA RED LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dense(64, activation='relu'))
model.add(Dense(acciones.shape[0], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. ENTRENAMIENTO
print("\nEntrenando la IA... (Esto tomará unos segundos)")
model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test))

# 5. GUARDADO DINÁMICO DEL MODELO
os.makedirs(os.path.join('..', 'models'), exist_ok=True)
# Aquí ocurre la magia: El archivo toma el nombre exacto de la categoría
nombre_archivo = f'modelo_{categoria}.h5'
ruta_guardado = os.path.join('..', 'models', nombre_archivo)

model.save(ruta_guardado)
print(f"\n✅ ¡Entrenamiento completado! El cerebro ha sido guardado como '{nombre_archivo}'")