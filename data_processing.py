"""
Módulo para preprocesamiento y generación de datos para el modelo CNN.
Este módulo contiene funciones reutilizables para crear generadores de datos.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from PIL import Image, ImageFile
import os

# Importar configuración
from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, 
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE, 
    HORIZONTAL_FLIP, VALIDATION_SPLIT
)

# IMPORTANTE: Configurar PIL para manejar imágenes truncadas
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
    
    # Imprimir información sobre los generadores
    print("\nEstadísticas de los generadores de datos:")
    print(f"- Vector de codificación de clases: {train_generator.class_indices}")
    print(f"- Entrenamiento: {train_generator.samples} instancias")
    print(f"- Pasos por época (train): {len(train_generator)} iteraciones")
    print(f"- Validación: {validation_generator.samples} instancias")
    print(f"- Pasos por época (val): {len(validation_generator)} iteraciones")
    print(f"- Test: {test_generator.samples} instancias")
    print(f"- Pasos para evaluación: {len(test_generator)} iteraciones")
    
    return train_generator, validation_generator, test_generator