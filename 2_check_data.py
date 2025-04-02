import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np

# Rutas a mis carpetas donde ya hice la separación de los datos.
TRAIN_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/train"
TEST_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/test"

def explore_dataset(base_dir):
    print(f"\nExplorando directorio: {base_dir}")
    
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Clases encontradas: {classes}")
    
    # Conteo de imágenes por clase
    stats = {}
    total_images = 0
    
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg',))]
        
        num_images = len(images)
        stats[cls] = num_images
        total_images += num_images
        
        # Obtener información de dimensiones de algunas imágenes, esto solo para mostrar resultados en el reporte
        if num_images > 0:
            sample_imgs = random.sample(images, min(5, num_images))
            dimensions = []
            
            for img_file in sample_imgs:
                img_path = os.path.join(class_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        dimensions.append(img.size)
                except Exception as e:
                    print(f"Error al abrir {img_path}: {e}")
            
            if dimensions:
                avg_width = sum(d[0] for d in dimensions) / len(dimensions)
                avg_height = sum(d[1] for d in dimensions) / len(dimensions)
                print(f"  - Clase '{cls}': {num_images} imágenes, dimensiones promedio: {avg_width:.1f}x{avg_height:.1f}")
            else:
                print(f"  - Clase '{cls}': {num_images} imágenes")
    
    print(f"Total de imágenes: {total_images}")
    return stats

print("=== EXPLORACIÓN DEL CONJUNTO DE DATOS ===")

# Datos de entrenamiento
train_stats = explore_dataset(TRAIN_DIR)

# Datos de prueba
test_stats = explore_dataset(TEST_DIR)

# Distribución de clases
if train_stats and test_stats:
    
    # Crear gráfico de barras comparativo
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_classes = sorted(set(list(train_stats.keys()) + list(test_stats.keys())))
    
    train_counts = [train_stats.get(cls, 0) for cls in all_classes]
    test_counts = [test_stats.get(cls, 0) for cls in all_classes]
    
    x = np.arange(len(all_classes))
    width = 0.35
    
    ax.bar(x - width/2, train_counts, width, label='Train')
    ax.bar(x + width/2, test_counts, width, label='Test')
    
    ax.set_title('Distribución de imágenes por clase')
    ax.set_xlabel('Clase')
    ax.set_ylabel('Número de imágenes')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes)
    ax.legend()
    
    for i, v in enumerate(train_counts):
        ax.text(i - width/2, v + 100, str(v), ha='center')
    
    for i, v in enumerate(test_counts):
        ax.text(i + width/2, v + 100, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
