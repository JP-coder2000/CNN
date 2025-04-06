# Detector de Imágenes Fake vs Real con CNN

Este proyecto implementa un clasificador binario de imágenes utilizando Redes Neuronales Convolucionales (CNN) para distinguir entre imágenes reales y falsas (generadas por IA). El modelo utiliza técnicas de transfer learning a través de la arquitectura ResNet50 preentrenada para maximizar el rendimiento con menor tiempo de entrenamiento.

## Estructura del Proyecto

El proyecto está organizado en scripts modulares que conforman un pipeline completo de machine learning:

1. **1_check_gpu.py**: Verifica la disponibilidad de GPU () y configuración de TensorFlow.
2. **2_check_data.py**: Explora el conjunto de datos y muestra estadísticas.
3. **3_data_processing.py**: Preprocesa las imágenes y configura los data generators.
4. **4_model.py**: Define la arquitectura del modelo CNN con transfer learning.
5. **5_training.py**: Entrena el modelo y evalúa su rendimiento.
6. **check_corrupted_images.py**: Script de utilidad para identificar y eliminar imágenes corruptas.

## Requisitos

- Python 3.8+ (de preferencia y por experiencia principal, recomiendo la versión 3.11.8)
- TensorFlow 2.x
- Pillow (PIL)
- NumPy
- Matplotlib

## Preparación del Entorno

1. Se recomienda usar un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate 
```

2. Instalar dependencias:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Solución de Problemas con Imágenes Corruptas

Si encuentras errores como "image file is truncated", primero ejecuta el script para identificar imágenes corruptas:

```bash
python check_corrupted_images.py
```

Este script:
- Verificará todas las imágenes en los directorios de entrenamiento y prueba
- Identificará imágenes corruptas
- Te dará la opción de eliminarlas automáticamente

## Ejecución del Pipeline

Se recomienda ejecutar los scripts en el siguiente orden:

1. **Verificar recursos computacionales**:
```bash
python 1_check_gpu.py
```

2. **Analizar el conjunto de datos**:
```bash
python 2_check_data.py
```

3. **Preparar los datos para entrenamiento**:
```bash
python 3_data_processing.py
```

4. **Crear la arquitectura del modelo**:
```bash
python 4_model.py
```

5. **Entrenar y evaluar el modelo**:
```bash
python 5_training.py
```

## Parámetros del Modelo

- **Arquitectura**: ResNet50 preentrenada en ImageNet (transfer learning)
- **Input**: Imágenes RGB de 224x224 píxeles
- **Optimizador**: Adam con learning rate adaptativo
- **Función de pérdida**: Binary Cross-Entropy
- **Métricas**: Accuracy, AUC-ROC, Precision, Recall, F1-Score
- **Regularización**: Dropout (0.5), Early Stopping, Reducción de learning rate

## Técnicas de Data Augmentation

Para mejorar la robustez del modelo, se aplican las siguientes transformaciones:
- Rotación aleatoria (±15°)
- Desplazamiento horizontal y vertical (10%)
- Volteo horizontal
- Normalización de píxeles al rango [0,1]

## Resultados y Visualización

Los resultados del entrenamiento se guardan en el directorio `resultados`:
- Modelo con mejores pesos (`cnn_detector_*_best.keras`)
- Modelo final (`cnn_detector_*_final.h5`)
- Curvas de aprendizaje (accuracy y loss)
- Historial de entrenamiento (formato JSON)

## Solución de Problemas Comunes

### Error: "image file is truncated"
Este error ocurre cuando hay imágenes corruptas en el conjunto de datos. Se puede solucionar:
1. Ejecutando `check_corrupted_images.py` para limpiar el dataset, o
2. Agregando las siguientes líneas al principio de tus scripts:
```python
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
```

### Errores de Memoria
Si experimentas errores de memoria, prueba:
- Reducir el tamaño del batch (`BATCH_SIZE`)
- Disminuir la resolución de las imágenes (`IMG_SIZE`)
- Usar `workers=1` y `use_multiprocessing=False` en `model.fit()`

## Notas Adicionales

- La validación cruzada se realiza mediante un split interno (80% train, 20% validation)
- El conjunto de test se mantiene separado para una evaluación imparcial
- Se implementan callbacks para evitar overfitting y guardar el mejor modelo
- Los logs de entrenamiento están disponibles en TensorBoard para visualización detallada