import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Configuración de hiperparámetros de entrenamiento
EPOCHS = 20              # Aumentamos a 20 para dar más tiempo de aprendizaje
BATCH_SIZE = 32          # Tamaño de mini-batch para SGD
IMG_SIZE = 224           # Dimensión espacial de entrada
PATIENCE = 5             # Épocas para early stopping
INITIAL_LR = 3e-4        # Aumentamos el learning rate inicial para convergencia más rápida
DROPOUT_RATE = 0.6       # Aumentamos el dropout para reducir overfitting

# Rutas a directorios
TRAIN_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/train"
TEST_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/test"
MODEL_DIR = "models"
LOGS_DIR = "logs"
RESULTS_DIR = "resultados"
BACKUP_DIR = "resultados/backups"  # Nueva carpeta para copias de seguridad

# Crear directorios si no existen
for dir_path in [MODEL_DIR, LOGS_DIR, RESULTS_DIR, BACKUP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Callback personalizado para hacer copias de seguridad
class BackupCallback(Callback):
    def __init__(self, checkpoint_filepath, backup_dir):
        super().__init__()
        self.checkpoint_filepath = checkpoint_filepath
        self.backup_dir = backup_dir
        
    def on_epoch_end(self, epoch, logs=None):
        # Hacer copia de seguridad cada 3 épocas
        if epoch % 3 == 0 and os.path.exists(self.checkpoint_filepath):
            backup_path = os.path.join(
                self.backup_dir, 
                f"model_backup_epoch_{epoch+1}.h5"
            )
            shutil.copy(self.checkpoint_filepath, backup_path)
            print(f"\nCreada copia de seguridad en: {backup_path}")

# Crear un nuevo modelo mejorado
def crear_modelo_cnn_mejorado():
    """
    Crea una arquitectura CNN mejorada con más regularización
    """
    print("\nCreando modelo CNN mejorado...")
    
    # Cargar modelo base preentrenado (backbone) sin capas de clasificación
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, models, optimizers
    
    modelo_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congelar capas inferiores del backbone
    for layer in modelo_base.layers[:-10]:  # Descongelamos las últimas 10 capas para fine-tuning
        layer.trainable = False
    
    # Construir arquitectura completa con más regularización
    modelo = models.Sequential([
        modelo_base,
        
        layers.GlobalAveragePooling2D(),
        
        # Primera capa densa con batch normalization y dropout
        layers.Dense(512, activation=None),  # Sin activación antes de batch norm
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Segunda capa densa con batch normalization y dropout
        layers.Dense(256, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Capa de salida para clasificación binaria
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilar modelo con optimizador adaptativo
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return modelo

# Decidir si crear un nuevo modelo o cargar uno existente
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_inicial.keras")
try:
    # Intentar cargar el modelo existente
    modelo = load_model(MODEL_PATH)
    print(f"Modelo cargado desde {MODEL_PATH}")
    
    # Aunque cargamos el modelo, lo vamos a recompilar con los nuevos parámetros
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
except:
    print(f"Creando nuevo modelo mejorado...")
    modelo = crear_modelo_cnn_mejorado()

# Configurar pipeline de datos con augmentation mejorado
print("\nConfigurando pipeline de datos con augmentation mejorado...")

# Preprocesamiento y data augmentation para entrenamiento (aumentamos la intensidad)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       # Aumentado de 15 a 20
    width_shift_range=0.2,   # Aumentado de 0.1 a 0.2
    height_shift_range=0.2,  # Aumentado de 0.1 a 0.2
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
print("\nConfigurando callbacks mejorados...")

# Nombre de modelo basado en timestamp para seguimiento de experimentos
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_name = f"cnn_detector_{timestamp}"

# Ruta para el mejor modelo
best_model_path = os.path.join(RESULTS_DIR, f'{model_name}_best.h5')

callbacks = [
    # Guardar modelo con mejor rendimiento en validación (criterio: accuracy)
    ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Guardar también el modelo con mejor pérdida (puede ser diferente al de mejor accuracy)
    ModelCheckpoint(
        filepath=os.path.join(RESULTS_DIR, f'{model_name}_best_loss.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    
    # Early stopping para detener entrenamiento cuando performance se estanca
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Guardar copia de seguridad periódicamente
    BackupCallback(best_model_path, BACKUP_DIR),
    
    # Reducir learning rate cuando el aprendizaje se estanca
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,       # Reduce el LR a la mitad
        patience=3,       # Espera 3 épocas antes de reducir
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
print("\nIniciando entrenamiento mejorado...")
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
print(f"- F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-10):.4f}")

# Visualización de curvas de aprendizaje
plt.figure(figsize=(14, 8))

# Gráfico de accuracy 
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy durante entrenamiento')
plt.ylabel('Accuracy')
plt.xlabel('Época')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Gráfico de función de pérdida
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss durante entrenamiento')
plt.ylabel('Binary Cross-Entropy')
plt.xlabel('Época')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

# Gráfico de AUC-ROC
plt.subplot(2, 2, 3)
plt.plot(history.history['auc'], label='Training')
plt.plot(history.history['val_auc'], label='Validation')
plt.title('AUC-ROC durante entrenamiento')
plt.ylabel('AUC-ROC')
plt.xlabel('Época')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Gráfico de learning rate
plt.subplot(2, 2, 4)
# Extraer learning rate si está disponible, o usar un valor constante si no
if 'lr' in history.history:
    plt.plot(history.history['lr'])
    plt.title('Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('Época')
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    plt.grid(True, linestyle='--', alpha=0.6)
else:
    # Mostrar Precision y Recall si LR no está disponible
    plt.plot(history.history['precision'], label='Precision (Train)')
    plt.plot(history.history['val_precision'], label='Precision (Val)')
    plt.plot(history.history['recall'], label='Recall (Train)')
    plt.plot(history.history['val_recall'], label='Recall (Val)')
    plt.title('Precision y Recall')
    plt.ylabel('Valor')
    plt.xlabel('Época')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_learning_curves.png'), dpi=300)
plt.show()

# Guardar modelo final
final_model_path = os.path.join(RESULTS_DIR, f'{model_name}_final.h5')
modelo.save(final_model_path)
print(f"\nModelo final guardado en: {final_model_path}")

# Crear una copia de seguridad del modelo final
backup_final_path = os.path.join(BACKUP_DIR, f'{model_name}_final.h5')
shutil.copy(final_model_path, backup_final_path)
print(f"Copia de seguridad del modelo final guardada en: {backup_final_path}")

# Guardar historial para análisis posterior
import json
with open(os.path.join(RESULTS_DIR, f'{model_name}_history.json'), 'w') as f:
    json.dump(history.history, f)
with open(os.path.join(BACKUP_DIR, f'{model_name}_history.json'), 'w') as f:
    json.dump(history.history, f)

print("\nProceso de entrenamiento y evaluación finalizado con éxito.")
print(f"Resultados guardados en directorio: {RESULTS_DIR}")
print(f"Copias de seguridad guardadas en: {BACKUP_DIR}")