import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import json  # Para guardar estadísticas para uso posterior

# Configuración
from config import TRAIN_DIR, TEST_DIR, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

def explore_dataset(base_dir):
    print(f"\nExplorando directorio: {base_dir}")
    
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Clases encontradas: {classes}")
    
    # Conteo de imágenes por clase
    stats = {}
    total_images = 0
    dimensions_by_class = {}
    
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        num_images = len(images)
        stats[cls] = num_images
        total_images += num_images
        
        # Obtener información de dimensiones de algunas imágenes
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
                dimensions_by_class[cls] = {"avg_width": avg_width, "avg_height": avg_height}
                print(f"  - Clase '{cls}': {num_images} imágenes, dimensiones promedio: {avg_width:.1f}x{avg_height:.1f}")
            else:
                print(f"  - Clase '{cls}': {num_images} imágenes")
    
    print(f"Total de imágenes: {total_images}")
    
    # Incluir información de dimensiones en las estadísticas
    stats_complete = {
        "counts": stats,
        "dimensions": dimensions_by_class,
        "total_images": total_images,
        "classes": classes
    }
    
    return stats_complete

print("=== EXPLORACIÓN DEL CONJUNTO DE DATOS ===")

# Datos de entrenamiento
train_stats = explore_dataset(TRAIN_DIR)

# Datos de prueba
test_stats = explore_dataset(TEST_DIR)

# Guardar estadísticas para uso posterior
dataset_stats = {
    "train": train_stats,
    "test": test_stats
}

# Guardar las estadísticas como JSON para referencia futura
stats_path = os.path.join(RESULTS_DIR, "dataset_stats.json")
with open(stats_path, 'w') as f:
    json.dump(dataset_stats, f, indent=4)
print(f"\nEstadísticas guardadas en: {stats_path}")

# Distribución de clases
if train_stats and test_stats:
    
    # Crear gráfico de barras comparativo
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_classes = sorted(set(train_stats["classes"] + test_stats["classes"]))
    
    train_counts = [train_stats["counts"].get(cls, 0) for cls in all_classes]
    test_counts = [test_stats["counts"].get(cls, 0) for cls in all_classes]
    
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
    plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=300)
    plt.show()