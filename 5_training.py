import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import shutil
import json
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback #type: ignore
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns

# Importar configuración
from config import (
    TRAIN_DIR, TEST_DIR, MODEL_PATH, BEST_MODEL_PATH, 
    LOGS_DIR, RESULTS_DIR, BACKUP_DIR, MODEL_NAME,
    EPOCHS, BATCH_SIZE, PATIENCE, CLASS_NAMES
)
# Importar funciones de preprocesamiento
from data_processing import crear_generadores_datos

# Configurar PIL para manejar imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Callback personalizado para hacer copias de seguridad
class BackupCallback(Callback):
    """Callback para hacer copias de seguridad del modelo durante el entrenamiento"""
    def __init__(self, checkpoint_filepath, backup_dir):
        super().__init__()
        self.checkpoint_filepath = checkpoint_filepath
        self.backup_dir = backup_dir
        
    def on_epoch_end(self, epoch, logs=None):
        # Hacer copia de seguridad cada 3 épocas
        if epoch % 3 == 0 and os.path.exists(self.checkpoint_filepath):
            backup_path = os.path.join(
                self.backup_dir, 
                f"model_backup_epoch_{epoch+1}.keras"
            )
            shutil.copy(self.checkpoint_filepath, backup_path)
            print(f"\nCreada copia de seguridad en: {backup_path}")

def crear_callbacks():
    """
    Configura los callbacks para el entrenamiento del modelo
    
    Returns:
        list: Lista de callbacks configurados
    """
    print("\nConfigurando callbacks para monitoreo y regularización...")
    
    # Ruta para guardar el mejor modelo
    best_model_path = BEST_MODEL_PATH
    
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
            filepath=os.path.join(RESULTS_DIR, f'{MODEL_NAME}_best_loss.keras'),
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
            log_dir=os.path.join(LOGS_DIR, MODEL_NAME),
            histogram_freq=1
        )
    ]
    
    return callbacks

def entrenar_modelo(modelo, train_generator, validation_generator):
    """
    Entrena el modelo utilizando los generadores de datos proporcionados
    
    Args:
        modelo: Modelo Keras a entrenar
        train_generator: Generador de datos de entrenamiento
        validation_generator: Generador de datos de validación
        
    Returns:
        history: Historial de entrenamiento
    """
    # Configurar callbacks
    callbacks = crear_callbacks()
    
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
    
    return history

def visualizar_curvas_aprendizaje(history):
    """
    Genera y guarda gráficos de las curvas de aprendizaje
    
    Args:
        history: Historial de entrenamiento del modelo
    """
    print("\nGenerando visualizaciones de curvas de aprendizaje...")
    
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

    # Gráfico de learning rate o Precision/Recall
    plt.subplot(2, 2, 4)
    # Extraer learning rate si está disponible, o usar Precision/Recall si no
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
    plt.savefig(os.path.join(RESULTS_DIR, f'{MODEL_NAME}_learning_curves.png'), dpi=300)
    plt.close()
    
    print(f"Curvas de aprendizaje guardadas en: {os.path.join(RESULTS_DIR, f'{MODEL_NAME}_learning_curves.png')}")

def generar_predicciones(modelo, test_generator):
    """
    Genera predicciones utilizando el modelo en el conjunto de test
    
    Args:
        modelo: Modelo Keras entrenado
        test_generator: Generador de datos de prueba
        
    Returns:
        tuple: (y_true, y_pred, y_pred_proba)
    """
    print("\nGenerando predicciones en conjunto de test...")
    
    # Reiniciar el generador para asegurar que comience desde el principio
    test_generator.reset()
    
    # Obtener las etiquetas verdaderas
    y_true = test_generator.classes
    
    # Generar predicciones (probabilidades)
    y_pred_proba = modelo.predict(test_generator, verbose=1)
    
    # Convertir probabilidades a etiquetas binarias
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    return y_true, y_pred, y_pred_proba.flatten()

def evaluar_modelo(modelo, test_generator, y_true, y_pred, y_pred_proba):
    """
    Evalúa el modelo y genera métricas y visualizaciones
    
    Args:
        modelo: Modelo Keras entrenado
        test_generator: Generador de datos de prueba
        y_true: Etiquetas verdaderas
        y_pred: Predicciones (etiquetas)
        y_pred_proba: Predicciones (probabilidades)
    """
    print("\nEvaluando rendimiento del modelo...")
    
    # Evaluación del modelo en conjunto de test
    test_loss, test_accuracy, test_auc, test_precision, test_recall = modelo.evaluate(test_generator)

    # Calcular F1-Score
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-10)

    print("\nMétricas de evaluación:")
    print(f"- Loss (BCE): {test_loss:.4f}")
    print(f"- Accuracy: {test_accuracy:.4f}")
    print(f"- AUC-ROC: {test_auc:.4f}")
    print(f"- Precision: {test_precision:.4f}")
    print(f"- Recall: {test_recall:.4f}")
    print(f"- F1-Score: {f1_score:.4f}")
    
    # Guardar métricas para referencia futura
    metrics = {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "auc": float(test_auc),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1_score": float(f1_score)
    }
    
    with open(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Generar reporte de clasificación
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    
    # Guardar reporte como JSON
    with open(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    
    # Imprimir reporte en texto
    print("\nReporte de clasificación detallado:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Generar y guardar matriz de confusión
    generar_matriz_confusion(y_true, y_pred)
    
    # Generar y guardar curva ROC
    generar_curva_roc(y_true, y_pred_proba)
    
    # Generar y guardar curva Precision-Recall
    generar_curva_precision_recall(y_true, y_pred_proba)

def generar_matriz_confusion(y_true, y_pred):
    """
    Genera y guarda la matriz de confusión
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones (etiquetas)
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    print(f"Matriz de confusión guardada en: {os.path.join(RESULTS_DIR, f'{MODEL_NAME}_confusion_matrix.png')}")
    
    # Calcular métricas específicas de la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    
    # Calcular métricas adicionales
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Guardar métricas adicionales
    additional_metrics = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "specificity": float(specificity)
    }
    
    with open(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_additional_metrics.json"), "w") as f:
        json.dump(additional_metrics, f, indent=4)

def generar_curva_roc(y_true, y_pred_proba):
    """
    Genera y guarda la curva ROC
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred_proba: Predicciones (probabilidades)
    """
    # Calcular puntos de la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Visualizar curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_roc_curve.png"), dpi=300)
    plt.close()
    
    print(f"Curva ROC guardada en: {os.path.join(RESULTS_DIR, f'{MODEL_NAME}_roc_curve.png')}")

def generar_curva_precision_recall(y_true, y_pred_proba):
    """
    Genera y guarda la curva Precision-Recall
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred_proba: Predicciones (probabilidades)
    """
    # Calcular puntos de la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Visualizar curva Precision-Recall
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'AUC = {pr_auc:.4f}')
    plt.axhline(y=sum(y_true)/len(y_true), color='navy', linestyle='--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_pr_curve.png"), dpi=300)
    plt.close()
    
    print(f"Curva Precision-Recall guardada en: {os.path.join(RESULTS_DIR, f'{MODEL_NAME}_pr_curve.png')}")

def guardar_historial(history):
    """
    Guarda el historial de entrenamiento para análisis posterior
    
    Args:
        history: Historial de entrenamiento del modelo
    """
    # Guardar historial como JSON
    with open(os.path.join(RESULTS_DIR, f'{MODEL_NAME}_history.json'), 'w') as f:
        # Convertir valores numpy a tipo Python estándar para poder serializar
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        json.dump(history_dict, f, indent=4)
    
    # Guardar copia de seguridad
    with open(os.path.join(BACKUP_DIR, f'{MODEL_NAME}_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"Historial de entrenamiento guardado en: {os.path.join(RESULTS_DIR, f'{MODEL_NAME}_history.json')}")

def main():
    """Función principal que ejecuta todo el proceso de entrenamiento y evaluación"""
    print("=== ENTRENAMIENTO Y EVALUACIÓN DEL MODELO ===")
    
    # Paso 1: Crear generadores de datos
    print("Paso 1: Configurando generadores de datos...")
    train_generator, validation_generator, test_generator = crear_generadores_datos()
    
    # Paso 2: Cargar modelo preconfigurado
    print("\nPaso 2: Cargando modelo...")
    try:
        modelo = load_model(MODEL_PATH)
        print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de haber ejecutado primero '4_model.py'")
        return
    
    # Paso 3: Entrenar modelo
    print("\nPaso 3: Entrenando modelo...")
    history = entrenar_modelo(modelo, train_generator, validation_generator)
    
    # Paso 4: Visualizar curvas de aprendizaje
    print("\nPaso 4: Visualizando curvas de aprendizaje...")
    visualizar_curvas_aprendizaje(history)
    
    # Paso 5: Generar predicciones
    print("\nPaso 5: Generando predicciones...")
    y_true, y_pred, y_pred_proba = generar_predicciones(modelo, test_generator)
    
    # Paso 6: Evaluar modelo
    print("\nPaso 6: Evaluando modelo...")
    evaluar_modelo(modelo, test_generator, y_true, y_pred, y_pred_proba)
    
    # Paso 7: Guardar historial y modelo final
    print("\nPaso 7: Guardando resultados finales...")
    guardar_historial(history)
    
    # Guardar modelo final
    final_model_path = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_final.keras')
    modelo.save(final_model_path)
    print(f"Modelo final guardado en: {final_model_path}")
    
    # Crear una copia de seguridad del modelo final
    backup_final_path = os.path.join(BACKUP_DIR, f'{MODEL_NAME}_final.keras')
    shutil.copy(final_model_path, backup_final_path)
    print(f"Copia de seguridad del modelo final guardada en: {backup_final_path}")
    
    print("\nProceso de entrenamiento y evaluación finalizado con éxito.")
    print(f"Resultados guardados en directorio: {RESULTS_DIR}")
    print(f"Copias de seguridad guardadas en: {BACKUP_DIR}")

if __name__ == "__main__":
    main()