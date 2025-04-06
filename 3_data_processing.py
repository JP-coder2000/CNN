import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

# IMPORTANTE: Configurar PIL para manejar imágenes truncadas (error que me daba al momento de correr el enrenamiento del modelo)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Aumentar el límite de píxeles para evitar DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

TRAIN_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/train"
TEST_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/test"

# Hiperparámetros de preprocesamiento
IMG_SIZE = 224 
BATCH_SIZE = 64 # Esto el primer run fue con 64, lo estoy pensando en ajustar en 32 y dejarlo corriendo.

# Configuración del pipeline de preprocesamiento con técnicas de data augmentation
train_datagen = ImageDataGenerator(
    # Normalización min-max para escalar valores de pixeles al rango [0,1]
    rescale=1./255,  
    
    # Técnicas de data augmentation
    rotation_range=15,       # rotación aleatoria (en grados)
    width_shift_range=0.1,   # rotación horizontal
    height_shift_range=0.1,  # Rotación vertical
    horizontal_flip=True,    # Horizontal Flip
    
    
    # cross-validation interna
    validation_split=0.2     # Split de datos para entrenamiento y validación
)

# Para el conjunto de test solo aplicamos normalización
test_datagen = ImageDataGenerator(rescale=1./255)

# Creación de generadores para entrenamiento basado en mini-batch
print("\nConfigurando generadores de datos para entrenamiento iterativo...")

# Generador para el subset de entrenamiento con data augmentation
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  # Redimensionamiento para entrada homogénea
    batch_size=BATCH_SIZE,             # Tamaño de mini-batch para SGD
    class_mode='binary',               # Modo binario para clasificación binaria
    subset='training'                  # Selector de subset para training
)

# Generador para validación interna durante entrenamiento (early stopping)
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'                # Selector de subset para validation
)

# Generador para evaluación final sobre conjunto de test
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False               # No mezclar datos para evaluación final
)

# Análisis de la configuración resultante
print("\nEstadísticas de los generadores de datos:")
print(f"- Vector de codificación de clases: {train_generator.class_indices}")
print(f"- Entrenamiento: {train_generator.samples} instancias")
print(f"- Pasos por época (train): {len(train_generator)} iteraciones")
print(f"- Validación: {validation_generator.samples} instancias")
print(f"- Pasos por época (val): {len(validation_generator)} iteraciones")
print(f"- Test: {test_generator.samples} instancias")
print(f"- Pasos para evaluación: {len(test_generator)} iteraciones")

# Función para capturar un lote de manera segura, con manejo de errores, esto me toco hacerlo porque el generador de datos a veces me daba error al momento de correr el modelo.
def get_safe_batch(generator, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return next(generator)
        except Exception as e:
            print(f"Error al obtener batch (intento {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                print("No se pudo obtener un batch válido después de varios intentos.")
                # Devolver arrays vacíos como fallback
                return np.zeros((0, IMG_SIZE, IMG_SIZE, 3)), np.zeros((0,))
    
# Inspección de la estructura de datos resultante
try:
    batch_x, batch_y = get_safe_batch(train_generator)
    
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
        plt.show()
    else:
        print("\nNo se pudieron obtener ejemplos para visualización.")
except Exception as e:
    print(f"\nError al visualizar ejemplos: {e}")

print("\nPipeline de preprocesamiento configurado exitosamente.")
print("Nota: Los tensores serán generados dinámicamente durante entrenamiento para optimizar memoria.")