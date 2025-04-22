import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import time

# Función para cargar y preprocesar una imagen
def preprocesar_imagen(ruta_imagen, tamaño_destino=(224, 224)):
    """
    Carga y preprocesa una imagen para inferencia con el modelo
    
    Args:
        ruta_imagen: Ruta a la imagen a procesar
        tamaño_destino: Dimensiones a las que redimensionar la imagen
        
    Returns:
        Tensor de la imagen preprocesada lista para inferencia
    """
    # Cargar imagen
    try:
        imagen = Image.open(ruta_imagen)
        # Convertir a RGB (por si es RGBA o escala de grises)
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        # Redimensionar
        imagen = imagen.resize(tamaño_destino)
        
        # Convertir a array y normalizar [0,1]
        imagen_array = np.array(imagen) / 255.0
        
        # Expandir dimensiones para batch
        imagen_tensor = np.expand_dims(imagen_array, axis=0)
        
        return imagen_tensor, imagen
    
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None, None

def mostrar_prediccion(imagen, prediccion, tiempo_inferencia):
    """
    Muestra la imagen con la predicción superpuesta
    
    Args:
        imagen: Imagen original
        prediccion: Valor de predicción [0-1]
        tiempo_inferencia: Tiempo que tomó la inferencia
    """
    # Crear figura
    plt.figure(figsize=(8, 8))
    plt.imshow(imagen)
    
    # Determinar predicción (0 = falsa, 1 = real)
    es_real = prediccion > 0.5
    etiqueta = "REAL" if es_real else "FALSA"
    confianza = prediccion if es_real else (1 - prediccion)
    
    # Color de la etiqueta (verde para real, rojo para falsa)
    color = "green" if es_real else "red"
    
    # Añadir texto con predicción
    plt.title(f"Predicción: {etiqueta}", fontsize=18, color=color)
    plt.xlabel(f"Confianza: {confianza*100:.2f}%\nTiempo: {tiempo_inferencia*1000:.1f}ms", fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    # Añadir borde de color según la predicción
    plt.gca().spines['top'].set_color(color)
    plt.gca().spines['bottom'].set_color(color)
    plt.gca().spines['left'].set_color(color)
    plt.gca().spines['right'].set_color(color)
    plt.gca().spines['top'].set_linewidth(5)
    plt.gca().spines['bottom'].set_linewidth(5)
    plt.gca().spines['left'].set_linewidth(5)
    plt.gca().spines['right'].set_linewidth(5)
    
    plt.tight_layout()
    plt.show()

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Detección de imágenes reales vs falsas con modelo CNN")
    parser.add_argument("--imagen", required=True, help="Ruta a la imagen para analizar")
    parser.add_argument("--modelo", default="resultados/cnn_detector_best.keras", 
                        help="Ruta al modelo entrenado")
    args = parser.parse_args()
    
    # Verificar existencia de archivos
    if not os.path.exists(args.imagen):
        print(f"Error: No se encontró la imagen en {args.imagen}")
        return
    
    if not os.path.exists(args.modelo):
        print(f"Error: No se encontró el modelo en {args.modelo}")
        return
    
    # Cargar modelo
    print(f"Cargando modelo desde: {args.modelo}")
    try:
        modelo = tf.keras.models.load_model(args.modelo)
        print("Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    # Procesar imagen
    print(f"Procesando imagen: {args.imagen}")
    imagen_tensor, imagen_original = preprocesar_imagen(args.imagen)
    
    if imagen_tensor is None:
        return
    
    # Realizar predicción y medir tiempo
    print("Realizando predicción...")
    inicio = time.time()
    
    # Usar aceleración GPU si está disponible
    try:
        with tf.device('/GPU:0'):
            predicciones = modelo.predict(imagen_tensor, verbose=0)
    except Exception:
        predicciones = modelo.predict(imagen_tensor, verbose=0)
    
    fin = time.time()
    tiempo_inferencia = fin - inicio
    
    # Obtener valor de predicción [0-1]
    prediccion = predicciones[0][0]
    
    # Imprimir resultados
    es_real = prediccion > 0.5
    etiqueta = "REAL" if es_real else "FALSA"
    confianza = prediccion if es_real else (1 - prediccion)
    
    print("\n" + "="*50)
    print(f"RESULTADO: La imagen es probablemente {etiqueta}")
    print(f"Confianza: {confianza*100:.2f}%")
    print(f"Valor crudo: {prediccion:.6f}")
    print(f"Tiempo de inferencia: {tiempo_inferencia*1000:.1f}ms")
    print("="*50)
    
    # Mostrar imagen con predicción
    mostrar_prediccion(imagen_original, prediccion, tiempo_inferencia)

if __name__ == "__main__":
    main()