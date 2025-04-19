import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import os
import json

# Importar configuración
from config import (
    TRAIN_DIR, TEST_DIR, RESULTS_DIR, 
    IMG_SIZE, BATCH_SIZE, 
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE, 
    HORIZONTAL_FLIP, VALIDATION_SPLIT
)
# Configurar PIL para manejar imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Aumentar el límite de píxeles para evitar DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

def crear_generadores_datos():
    """
    Crea y configura los generadores de datos para entrenamiento, validación y prueba.
    
    Returns:
        tuple: (train_generator, validation_generator, test_generator)
    """
    print("\nConfigurando generadores de datos para entrenamiento iterativo...")
    
    # Configuración del pipeline de preprocesamiento con técnicas de data augmentation
    train_datagen = ImageDataGenerator(
        # Normalización min-max para escalar valores de pixeles al rango [0,1]
        rescale=1./255,  
        
        # Técnicas de data augmentation
        rotation_range=ROTATION_RANGE,       # rotación aleatoria (en grados)
        width_shift_range=WIDTH_SHIFT_RANGE,   # rotación horizontal
        height_shift_range=HEIGHT_SHIFT_RANGE,  # rotación vertical
        horizontal_flip=HORIZONTAL_FLIP,    # Horizontal Flip
        
        # cross-validation interna
        validation_split=VALIDATION_SPLIT     # Split de datos para entrenamiento y validación
    )

    # Para el conjunto de test solo aplicamos normalización
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generador para el subset de entrenamiento con data augmentation
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),  # Redimensionamiento para entrada homogénea
        batch_size=BATCH_SIZE,             # Tamaño de mini-batch para SGD
        class_mode='binary',               # Modo binario para clasificación binaria
        subset='training',                 # Selector de subset para training
        shuffle=True                       # Mezclar datos para entrenamiento
    )

    # Generador para validación interna durante entrenamiento (early stopping)
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',              # Selector de subset para validation
        shuffle=True                      # Mezclar datos para validación
    )

    # Generador para evaluación final sobre conjunto de test
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False                     # No mezclar datos para evaluación final
    )
    
    return train_generator, validation_generator, test_generator

# Analizar y guardar información sobre los generadores de datos
def analizar_generadores(train_generator, validation_generator, test_generator):
    """
    Analiza y guarda información sobre los generadores de datos.
    
    Args:
        train_generator: Generador de datos de entrenamiento
        validation_generator: Generador de datos de validación
        test_generator: Generador de datos de prueba
    """
    # Análisis de la configuración resultante
    generators_info = {
        "class_indices": train_generator.class_indices,
        "train_samples": train_generator.samples,
        "train_steps_per_epoch": len(train_generator),
        "validation_samples": validation_generator.samples,
        "validation_steps": len(validation_generator),
        "test_samples": test_generator.samples,
        "test_steps": len(test_generator),
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE
    }
    
    # Guardar la información para uso en scripts posteriores
    with open(os.path.join(RESULTS_DIR, "generators_info.json"), "w") as f:
        json.dump(generators_info, f, indent=4)
    
    print("\nEstadísticas de los generadores de datos:")
    print(f"- Vector de codificación de clases: {train_generator.class_indices}")
    print(f"- Entrenamiento: {train_generator.samples} instancias")
    print(f"- Pasos por época (train): {len(train_generator)} iteraciones")
    print(f"- Validación: {validation_generator.samples} instancias")
    print(f"- Pasos por época (val): {len(validation_generator)} iteraciones")
    print(f"- Test: {test_generator.samples} instancias")
    print(f"- Pasos para evaluación: {len(test_generator)} iteraciones")
    
    return generators_info

# Función para obtener y visualizar un lote de manera segura
def visualizar_batch(train_generator, max_attempts=3):
    """
    Obtiene y visualiza un lote del generador de datos de forma segura.
    
    Args:
        train_generator: Generador de datos de entrenamiento
        max_attempts: Número máximo de intentos para obtener un lote
        
    Returns:
        tuple: (batch_x, batch_y) o (None, None) si hay error
    """
    for attempt in range(max_attempts):
        try:
            batch_x, batch_y = next(train_generator)
            
            if len(batch_x) > 0:
                print(f"\nEstructura de tensor de entrada: {batch_x.shape}")
                print(f"Estructura de tensor de etiquetas: {batch_y.shape}")
                print(f"Rango de valores tras normalización: [{batch_x.min()}, {batch_x.max()}]")

                # Visualización de ejemplos de data augmentation
                plt.figure(figsize=(12, 8))
                for i in range(min(9, batch_x.shape[0])):
                    plt.subplot(3, 3, i+1)
                    plt.imshow(batch_x[i])
                    plt.title(f"Clase: {'Real' if batch_y[i] > 0.5 else 'Fake'}")
                    plt.axis('off')

                plt.tight_layout()
                plt.suptitle("Ejemplos de imágenes preprocesadas con data augmentation", y=0.98)
                plt.savefig(os.path.join(RESULTS_DIR, "batch_examples.png"), dpi=300)
                plt.show()
                
                return batch_x, batch_y
        
        except Exception as e:
            print(f"Error al obtener batch (intento {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                print("No se pudieron obtener ejemplos para visualización.")
                return None, None

if __name__ == "__main__":
    # Crear y analizar generadores de datos
    train_generator, validation_generator, test_generator = crear_generadores_datos()
    
    # Analizar y guardar información sobre los generadores
    analizar_generadores(train_generator, validation_generator, test_generator)
    
    # Visualizar un lote de ejemplo
    visualizar_batch(train_generator)
    
    print("\nPipeline de preprocesamiento configurado exitosamente.")
    print("Nota: Los tensores serán generados dinámicamente durante entrenamiento para optimizar memoria.")