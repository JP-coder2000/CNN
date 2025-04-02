import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Definición de rutas a los conjuntos de datos
TRAIN_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/train"
TEST_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/test"

# Hiperparámetros de preprocesamiento
IMG_SIZE = 224 
BATCH_SIZE = 32

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

# Inspección de la estructura de datos resultante
batch_x, batch_y = next(train_generator)
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

print("\nPipeline de preprocesamiento configurado exitosamente.")
print("Nota: Los tensores serán generados dinámicamente durante entrenamiento para optimizar memoria.")