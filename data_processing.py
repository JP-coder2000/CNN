import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración de parámetros
TRAIN_DIR = "train"
TEST_DIR = "test"
# Defino este tamaño de imagen para que sea el mismo en todos los generadores
# y así no tener problemas de dimensiones al cargar las imágenes.
IMG_SIZE = 224 

# Tamaño del batch, esto es para experimentar
BATCH_SIZE = 64
SEED = 42  # Para reproducibilidad

# Crear carpeta para guardar imágenes aumentadas de ejemplo
SAMPLE_DIR = "muestras_aumentadas"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Definir generadores de datos
def create_data_generators(with_augmentation=True):
    """Crea generadores de datos para entrenamiento y prueba"""
    
    if with_augmentation:
        # Generador con aumentación de datos para entrenamiento
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalización
            rotation_range=20,  # Rotación aleatoria
            width_shift_range=0.2,  # Desplazamiento horizontal
            height_shift_range=0.2,  # Desplazamiento vertical
            shear_range=0.2,  # Transformación de corte
            zoom_range=0.2,  # Zoom aleatorio
            horizontal_flip=True,  # Volteo horizontal
            fill_mode='nearest',  # Modo de relleno para píxeles nuevos
            validation_split=0.2  # Separar 20% para validación
        )
    else:
        # Generador básico sin aumentación (solo normalización)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    
    # Generador para datos de prueba (solo normalización)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Crear generador de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Para clasificación binaria
        subset='training',
        seed=SEED
    )
    
    # Crear generador de validación
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=SEED
    )
    
    # Crear generador de prueba
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # No mezclar para mantener orden de archivos
    )
    
    return train_generator, validation_generator, test_generator

# Función para visualizar aumentación de datos
def visualize_data_augmentation():
    """Visualiza ejemplos de imágenes aumentadas"""
    
    # Crear un generador específico para visualización (con batch_size=1)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Cargar imágenes de entrenamiento
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    for cls in classes:
        # Crear generador para esta clase
        gen = datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=1,
            classes=[cls],  # Solo imágenes de esta clase
            shuffle=True,
            seed=SEED
        )
        
        # Obtener una imagen original
        batch = next(gen)
        image = batch[0][0]  # Primera imagen del batch
        
        # Crear figura para mostrar original y aumentaciones
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 3, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")
        
        # Generar y mostrar 8 versiones aumentadas
        for i in range(8):
            batch = next(gen)
            augmented_image = batch[0][0]
            
            plt.subplot(3, 3, i + 2)
            plt.imshow(augmented_image)
            plt.title(f"Aumentada {i+1}")
            plt.axis("off")
        
        # Guardar figura
        plt.tight_layout()
        plt.savefig(os.path.join(SAMPLE_DIR, f"aumentacion_{cls}.jpg"))
        plt.close()
        
        print(f"Imágenes aumentadas de clase '{cls}' guardadas en {SAMPLE_DIR}")

# Función para verificar un batch de imágenes
def check_batch(generators):
    """Verifica y muestra un batch de imágenes de cada generador"""
    
    train_gen, val_gen, test_gen = generators
    
    # Obtener mapeo de clases
    class_indices = train_gen.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    print("Mapeo de clases:", class_indices)
    
    # Función para mostrar batch
    def show_batch(generator, title):
        # Obtener un batch
        batch_x, batch_y = next(generator)
        
        # Configurar figura
        plt.figure(figsize=(15, 8))
        plt.suptitle(title, fontsize=16)
        
        # Mostrar hasta 12 imágenes del batch
        for i in range(min(12, batch_x.shape[0])):
            plt.subplot(3, 4, i + 1)
            
            # Convertir etiqueta numérica a nombre de clase
            label = batch_y[i]
            class_name = class_names[int(np.round(label))]
            
            plt.imshow(batch_x[i])
            plt.title(f"Clase: {class_name} ({label:.1f})")
            plt.axis("off")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para título principal
        return plt.gcf()  # Devolver la figura para guardarla
    
    # Mostrar y guardar batches
    train_fig = show_batch(train_gen, "Batch de Entrenamiento")
    train_fig.savefig(os.path.join(SAMPLE_DIR, "batch_entrenamiento.jpg"))
    plt.close(train_fig)
    
    val_fig = show_batch(val_gen, "Batch de Validación")
    val_fig.savefig(os.path.join(SAMPLE_DIR, "batch_validacion.jpg"))
    plt.close(val_fig)
    
    test_fig = show_batch(test_gen, "Batch de Prueba")
    test_fig.savefig(os.path.join(SAMPLE_DIR, "batch_prueba.jpg"))
    plt.close(test_fig)
    
    print(f"Ejemplos de batches guardados en {SAMPLE_DIR}")

# Ejecutar el script
if __name__ == "__main__":
    print("=== PREPROCESAMIENTO DE DATOS ===")
    
    # Crear generadores con aumentación
    print("\nCreando generadores de datos con aumentación...")
    generators = create_data_generators(with_augmentation=True)
    train_gen, val_gen, test_gen = generators
    
    # Verificar funcionamiento
    print("\nInformación del generador de entrenamiento:")
    print(f"- Número de clases: {len(train_gen.class_indices)}")
    print(f"- Nombres de clases: {list(train_gen.class_indices.keys())}")
    print(f"- Pasos por época: {len(train_gen)}")
    print(f"- Tamaño del batch: {train_gen.batch_size}")
    print(f"- Total de imágenes: {train_gen.samples}")
    
    print("\nInformación del generador de validación:")
    print(f"- Pasos por época: {len(val_gen)}")
    print(f"- Total de imágenes: {val_gen.samples}")
    
    print("\nInformación del generador de prueba:")
    print(f"- Pasos por época: {len(test_gen)}")
    print(f"- Total de imágenes: {test_gen.samples}")
    
    # Visualizar aumentación de datos
    print("\nGenerando visualizaciones de aumentación de datos...")
    visualize_data_augmentation()
    
    # Verificar un batch de cada generador
    print("\nVerificando y visualizando batches de cada generador...")
    check_batch(generators)
    
    print("\n✅ Preprocesamiento completado. Revisa las imágenes generadas en la carpeta", SAMPLE_DIR)