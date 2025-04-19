import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from keras import layers, models, optimizers
import os


# Hiperparámetros de la arquitectura mejorada
IMG_SIZE = 224     # Dimensión espacial para entrada de la red
LEARNING_RATE = 3e-4  # Aumentado para convergencia más rápida
DROPOUT_RATE = 0.5  # Aumentado para reducir overfitting

# Directorio para almacenar modelos
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

#Funcion principal para crear la arquitectura del modelo mejorada
def crear_modelo_cnn_mejorado():
    """
    Implementa una arquitectura CNN mediante transfer learning con ResNet50.
    Incluye mejoras para reducir overfitting y mejorar la generalización.
    
    Returns:
        Modelo compilado listo para entrenamiento
    """
    print("Inicializando arquitectura CNN mejorada con transfer learning...")
    
    # Cargar modelo base preentrenado (backbone) sin capas de clasificación
    modelo_base = ResNet50(
        weights='imagenet',    # Inicialización con pesos preentrenados en ImageNet
        include_top=False,     # Excluir capas fully-connected superiores
        input_shape=(IMG_SIZE, IMG_SIZE, 3)  # Dimensiones de entrada: HxWxC
    )
    
    # Congelar pesos del backbone para evitar catástrofe de olvido
    # Solo dejamos entrenar las últimas 15 capas para fine-tuning
    for layer in modelo_base.layers[:-15]:
        layer.trainable = False
        
    print(f"Base parcialmente congelada: {modelo_base.name}")
    print(f"Capas entrenables: {len([l for l in modelo_base.layers if l.trainable])}/{len(modelo_base.layers)}")
    
    # Construir arquitectura completa mejorada
    modelo = models.Sequential([
        modelo_base,
        
        layers.GlobalAveragePooling2D(),
        
        # Primera capa densa con batch normalization para estabilizar el entrenamiento
        layers.Dense(512, use_bias=False),  # Sin bias antes de batch norm
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),  # Fuerte dropout para evitar overfitting
        
        # Segunda capa densa también con regularización
        layers.Dense(256, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Capa de salida para clasificación binaria con activación sigmoide
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilar modelo definiendo función de pérdida, optimizador y métricas
    modelo.compile(
        # Optimizador adaptativo con learning rate dinámico
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        
        # Binary cross-entropy: función de pérdida estándar para clasificación binaria
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

# Crear la arquitectura del modelo mejorado
modelo = crear_modelo_cnn_mejorado()

# Visualizar resumen de la arquitectura
print("\nArquitectura del modelo CNN mejorado:")
modelo.summary()

# Guardar arquitectura inicial
modelo_path = os.path.join(MODEL_DIR, 'modelo_mejorado.keras')
modelo.save(modelo_path)

print(f"\nArquitectura mejorada guardada en: {modelo_path}")
print("\nPrincipales mejoras implementadas:")
print("1. Descongelamiento parcial de la red base para fine-tuning")
print("2. Adición de batch normalization para estabilizar el entrenamiento")
print("3. Aumento del dropout para reducir overfitting")
print("4. Estructura más profunda con capa intermedia adicional")
print("5. Learning rate optimizado para convergencia más rápida")