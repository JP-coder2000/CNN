# Detector de Imágenes Fake vs Real con CNN

Este proyecto implementa un clasificador binario basado en Redes Neuronales Convolucionales (CNN) que distingue entre imágenes reales y falsas (generadas por IA). Utilizando transfer learning con ResNet50, el modelo alcanza un accuracy de 77.3% y un AUC-ROC de 0.863.

## Características Principales

- **Arquitectura**: ResNet50 con fine-tuning selectivo (últimas 15 capas)
- **Preprocesamiento**: Data augmentation (rotación, desplazamiento, flip) y normalización
- **Regularización**: Dropout (0.6), Batch Normalization, Early Stopping
- **Optimización**: Adam con learning rate adaptativo (3e-4 inicial con reducción progresiva)

## Estructura del Proyecto

```
├── 1_check_gpu.py            # Verifica disponibilidad de GPU
├── 2_check_data.py           # Explora y analiza el dataset
├── 3_data_processing.py      # Preprocesa imágenes y configura generadores
├── 4_model.py                # Define la arquitectura CNN
├── 5_training.py             # Entrena y evalúa el modelo
├── check_corrupted_images.py # Identifica imágenes corruptas
├── config.py                 # Configuración global del proyecto
├── models/                   # Modelos guardados
└── resultados/               # Métricas, gráficos y análisis
```

## Requisitos

- Python 3.8+ (recomendado 3.11.8)
- TensorFlow 2.x
- NumPy, Matplotlib, Pillow
- scikit-learn (para métricas de evaluación)

## Instalación Rápida

```bash
python -m venv venv
source venv/bin/activate
pip install tensorflow numpy matplotlib pillow scikit-learn
```

## Uso

1. Ejecutar los scripts en orden numérico (1 → 5)
2. Para entrenar con GPU:
   ```bash
   python 5_training.py
   ```
3. Revisar resultados en el directorio `resultados/`

## Rendimiento

| Métrica      | Valor  |
|--------------|--------|
| Accuracy     | 77.3%  |
| AUC-ROC      | 0.863  |
| Precision    | 0.843  |
| Recall       | 0.671  |
| F1-Score     | 0.747  |
| Especificidad| 0.875  |

## Solución de Problemas

- Para errores con imágenes corruptas: `python check_corrupted_images.py`
- Para errores de memoria: reducir `BATCH_SIZE` o `IMG_SIZE` en `config.py`

## Basado en

Este proyecto toma como referencia el trabajo de Kroiß y Reschke "Deepfake Detection of Face Images based on a Convolutional Neural Network", adaptando su enfoque de transfer learning con ResNet50 para clasificación binaria de imágenes.