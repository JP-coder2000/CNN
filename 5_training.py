import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard # type: ignore
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Configuración de hiperparámetros de entrenamiento
EPOCHS = 10              # Número de iteraciones completas sobre el dataset
BATCH_SIZE = 32          # Tamaño de mini-batch para SGD
IMG_SIZE = 224           # Dimensión espacial de entrada
PATIENCE = 3             # Épocas para early stopping
INITIAL_LR = 1e-4        # Learning rate inicial

# Rutas a directorios
TRAIN_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/train"
TEST_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/test"
MODEL_DIR = "models"
LOGS_DIR = "logs"
RESULTS_DIR = "resultados"

# Cargar modelo previamente definido
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_inicial.keras")
try:
    modelo = load_model(MODEL_PATH)
    print(f"Modelo cargado desde {MODEL_PATH}")
except:
    print(f"Error: No se pudo cargar el modelo desde {MODEL_PATH}")
    print("Ejecute primero 4_model_definition.py")
    exit(1)

# Configurar pipeline de datos
print("\nConfigurando pipeline de datos...")

# Preprocesamiento y data augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Solo normalización para evaluación
test_datagen = ImageDataGenerator(rescale=1./255)

# Generador para mini-batches de entrenamiento
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Generador para validación durante entrenamiento
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Generador para evaluación final
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Definir callbacks para monitoreo y regularización del entrenamiento
print("\nConfigurando callbacks para monitoreo...")

# Nombre de modelo basado en timestamp para seguimiento de experimentos
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_name = f"cnn_detector_{timestamp}"

callbacks = [
    # Guardar modelo con mejor rendimiento en validación (criterio: accuracy)
    ModelCheckpoint(
        filepath=os.path.join(RESULTS_DIR, f'{model_name}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Early stopping para detener entrenamiento cuando performance se estanca
    # Previene overfitting y optimiza tiempo de entrenamiento
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reducir learning rate cuando el aprendizaje se estanca
    # Permite afinar en etapas finales del entrenamiento
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    
    # TensorBoard para visualización del proceso de entrenamiento
    TensorBoard(
        log_dir=os.path.join(LOGS_DIR, model_name),
        histogram_freq=1
    )
]

# Entrenamiento del modelo
print("\nIniciando entrenamiento...")
start_time = time.time()

# Usar aceleración GPU si está disponible
try:
    with tf.device('/GPU:0'):
        history = modelo.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        print("\n✓ Entrenamiento completado con aceleración GPU.")
except Exception as e:
    print(f"\n⚠ GPU no disponible o error: {e}")
    print("Utilizando CPU...")
    history = modelo.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    print("\n✓ Entrenamiento completado con CPU.")

# Calcular tiempo de entrenamiento
end_time = time.time()
training_time = end_time - start_time
hours, rem = divmod(training_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Tiempo de entrenamiento: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

# Evaluación del modelo en conjunto de test
print("\nEvaluando rendimiento en conjunto de test (datos no vistos)...")
test_loss, test_accuracy, test_auc, test_precision, test_recall = modelo.evaluate(test_generator)

print("\nMétricas de evaluación:")
print(f"- Loss (BCE): {test_loss:.4f}")
print(f"- Accuracy: {test_accuracy:.4f}")
print(f"- AUC-ROC: {test_auc:.4f}")
print(f"- Precision: {test_precision:.4f}")
print(f"- Recall: {test_recall:.4f}")
print(f"- F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# Visualización de curvas de aprendizaje
plt.figure(figsize=(12, 5))

# Gráfico de accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy durante entrenamiento')
plt.ylabel('Accuracy')
plt.xlabel('Época')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Gráfico de función de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss durante entrenamiento')
plt.ylabel('Binary Cross-Entropy')
plt.xlabel('Época')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_learning_curves.png'), dpi=300)
plt.show()

# Guardar modelo final
final_model_path = os.path.join(RESULTS_DIR, f'{model_name}_final.h5')
modelo.save(final_model_path)
print(f"\nModelo final guardado en: {final_model_path}")

# Guardar historial para análisis posterior
import json
with open(os.path.join(RESULTS_DIR, f'{model_name}_history.json'), 'w') as f:
    json.dump(history.history, f)

print("\nProceso de entrenamiento y evaluación finalizado con éxito.")
print(f"Resultados guardados en directorio: {RESULTS_DIR}")