import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from keras import layers, models, optimizers
import os


# Hiperparámetros de la arquitectura
IMG_SIZE = 224     # Dimensión espacial para entrada de la red
LEARNING_RATE = 1e-4  # Tasa de aprendizaje para optimizador Adam

# Directorio para almacenar modelos
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

#Funcion principal para crear la arquitectura del modelo
def crear_modelo_cnn():
    """
    Implementa una arquitectura CNN mediante transfer learning con ResNet50.
    
    Utiliza un enfoque de fine-tuning con dos fases:
    1. Feature extraction: congelando la red base preentrenada
    2. Fine-tuning: entrenamiento de capas superiores personalizadas
    
    Returns:
        Modelo compilado listo para entrenamiento
    """
    print("Inicializando arquitectura CNN con transfer learning...")
    
    # Cargar modelo base preentrenado (backbone) sin capas de clasificación
    modelo_base = ResNet50(
        weights='imagenet',    # Inicialización con pesos preentrenados en ImageNet
        include_top=False,     # Excluir capas fully-connected superiores
        input_shape=(IMG_SIZE, IMG_SIZE, 3)  # Dimensiones de entrada: HxWxC (la verdad esto lo tome de la documentación ofiical de keras)
    )
    
    # Congelar pesos del backbone para evitar catástrofe de olvido (esto me paso con mi proyecto de IA generativa)
    modelo_base.trainable = False
    print(f"Base congelada: {modelo_base.name} con {modelo_base.count_params():,} parámetros")
    
    # Construir arquitectura completa
    modelo = models.Sequential([
        modelo_base,
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'),
        
        # Regularización mediante dropout para prevenir overfitting
        layers.Dropout(0.5),
        
        # Capa de salida para clasificación binaria con activación sigmoide
        # La sigmoide comprime valores al rango [0,1] interpretable como probabilidad
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilar modelo definiendo función de pérdida, optimizador y métricas
    modelo.compile(
        # Optimizador adaptativo con learning rate dinámico
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        
        # Binary cross-entropy: función de pérdida estándar para clasificación binaria
        # Minimiza la divergencia entre distribuciones de probabilidad
        loss='binary_crossentropy',
        
        # Métricas para monitorizar durante entrenamiento y evaluación
        metrics=[
            'accuracy',  # Precisión global: (TP+TN)/(TP+TN+FP+FN)
            tf.keras.metrics.AUC(),  # Área bajo la curva ROC
            tf.keras.metrics.Precision(),  # Precisión: TP/(TP+FP)
            tf.keras.metrics.Recall()  # Recall (sensibilidad): TP/(TP+FN)
        ]
    )
    
    return modelo

# Crear la arquitectura del modelo
modelo = crear_modelo_cnn()

# Visualizar resumen de la arquitectura
print("\nArquitectura del modelo CNN:")
modelo.summary()

# Guardar arquitectura inicial
modelo_path = os.path.join(MODEL_DIR, 'modelo_inicial.keras')
modelo.save(modelo_path)

print(f"\nArquitectura base guardada en: {modelo_path}")