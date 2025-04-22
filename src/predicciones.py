import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('modelo_coca_vs_pepsi.h5')

# Nombres de clases (ajusta si hace falta)
clases = ['coca', 'pepsi']

def predecir_imagen(ruta_imagen):
    # Cargar y preparar la imagen
    img = image.load_img(ruta_imagen, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer predicción
    pred = modelo.predict(img_array)[0]
    clase_predicha = clases[np.argmax(pred)]
    confianza = np.max(pred) * 100

    # Mostrar imagen con predicción
    img_original = image.load_img(ruta_imagen)
    plt.imshow(img_original)
    plt.axis('off')
    
    texto = f'{clase_predicha} ({confianza:.2f}%)'
    plt.text(
        5, 15, texto,
        fontsize=12, color='white', backgroundcolor='black',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')
    )

    plt.show()

predecir_imagen('prueba12.jpg')
