import sys
import os

# Agregar carpeta 'src' al path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

# Importar la función desde predicciones
from predicciones import predecir_imagen

# Ruta de la imagen de prueba (ajústala según tu caso)
ruta_imagen = 'prueba1.jpg'  # Debe estar en la misma carpeta que main.py o pon la ruta completa

# Ejecutar la predicción
predecir_imagen(ruta_imagen)
