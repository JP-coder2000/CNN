import tensorflow as tf
from tensorflow.keras.applications import ResNet50 #type: ignore
from tensorflow.keras import layers, models, optimizers #type: ignore
import os
import json
import numpy as np

# Importar configuración
from config import (
    IMG_SIZE, LEARNING_RATE, DROPOUT_RATE, 
    MODEL_PATH, MODEL_DIR
)

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

if __name__ == "__main__":
    # Asegurar que el directorio existe
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Crear la arquitectura del modelo mejorado
    modelo = crear_modelo_cnn_mejorado()

    # Visualizar resumen de la arquitectura
    print("\nArquitectura del modelo CNN mejorado:")
    modelo.summary()
    
    # Intentar guardar el diagrama del modelo para referencia
    try:
        tf.keras.utils.plot_model(
            modelo, 
            to_file=os.path.join(MODEL_DIR, 'model_architecture.png'),
            show_shapes=True, 
            show_dtype=True, 
            show_layer_names=True
        )
        print(f"Diagrama del modelo guardado en: {os.path.join(MODEL_DIR, 'model_architecture.png')}")
    except ImportError:
        print("No se pudo generar el diagrama del modelo. Se requiere instalar graphviz correctamente.")
    
    # Guardar arquitectura inicial en formato moderno .keras
    modelo.save(MODEL_PATH)
    print(f"\nArquitectura mejorada guardada en: {MODEL_PATH}")
    
    # Guardar también la estructura como JSON para referencia
    trainable_params = int(sum(tf.keras.backend.count_params(p) for p in modelo.trainable_weights))
    non_trainable_params = int(sum(tf.keras.backend.count_params(p) for p in modelo.non_trainable_weights))
    
    model_config = {
        "num_layers": len(modelo.layers),
        "params": int(modelo.count_params()),
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }
    
    with open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    
    print("\nPrincipales características del modelo:")
    print("1. Transfer learning con ResNet50 preentrenado")
    print("2. Fine-tuning de las últimas capas para adaptación al problema específico")
    print("3. Batch normalization para estabilizar el entrenamiento")
    print(f"4. Alto dropout ({DROPOUT_RATE}) para reducir overfitting")
    print("5. Estructura multi-capa con regularización")