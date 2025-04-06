import os
from PIL import Image, ImageFile

# Configurar PIL para manejar imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Evitar advertencias de "decompression bomb"

# Rutas a los conjuntos de datos (usa tus rutas reales)
TRAIN_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/train"
TEST_DIR = "/Users/juanpablocabreraquiroga/Documents/Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales/test"

def verificar_imagenes(directorio_base):
    """
    Función para verificar y reportar imágenes corruptas en el conjunto de datos
    """
    print(f"Verificando imágenes en: {directorio_base}")
    
    imagenes_corruptas = []
    total_verificadas = 0
    
    clases = [d for d in os.listdir(directorio_base) if os.path.isdir(os.path.join(directorio_base, d))]
    
    for clase in clases:
        clase_dir = os.path.join(directorio_base, clase)
        archivos = [f for f in os.listdir(clase_dir) if os.path.isfile(os.path.join(clase_dir, f)) 
                  and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Verificando {len(archivos)} imágenes en clase '{clase}'...")
        
        for i, archivo in enumerate(archivos):
            ruta_img = os.path.join(clase_dir, archivo)
            try:
                # Intenta abrir y cargar la imagen para verificar si está corrupta
                with Image.open(ruta_img) as img:
                    # Cargar la imagen fuerza la lectura completa del archivo
                    img.load()
                
                total_verificadas += 1
                # Mostrar progreso cada 500 imágenes
                if total_verificadas % 500 == 0:
                    print(f"  Progreso: {total_verificadas} imágenes verificadas...")
                    
            except Exception as e:
                imagenes_corruptas.append((ruta_img, str(e)))
                print(f"  Imagen corrupta encontrada: {ruta_img}")
    
    # Reportar resultados
    if imagenes_corruptas:
        print(f"\nSe encontraron {len(imagenes_corruptas)} imágenes corruptas de un total de {total_verificadas}:")
        for i, (img, error) in enumerate(imagenes_corruptas[:20]):  # Mostrar solo las primeras 20
            print(f" - {img}: {error}")
        
        if len(imagenes_corruptas) > 20:
            print(f"  ... y {len(imagenes_corruptas) - 20} más.")
    else:
        print(f"\nNo se encontraron imágenes corruptas en {total_verificadas} imágenes.")
    
    return imagenes_corruptas

def eliminar_imagenes_corruptas(imagenes):
    """
    Elimina las imágenes corruptas del sistema de archivos
    """
    if not imagenes:
        return
    
    print(f"\nEliminar {len(imagenes)} imágenes corruptas? (s/n)")
    respuesta = input().strip().lower()
    
    if respuesta == 's':
        print(f"Eliminando {len(imagenes)} imágenes corruptas...")
        eliminadas = 0
        
        for img_path, _ in imagenes:
            try:
                os.remove(img_path)
                eliminadas += 1
                if eliminadas % 10 == 0:
                    print(f"  Progreso: {eliminadas}/{len(imagenes)} eliminadas...")
            except Exception as e:
                print(f"  Error al eliminar {img_path}: {e}")
        
        print(f"\n✅ Eliminadas {eliminadas} imágenes corruptas.")
    else:
        print("Operación cancelada por el usuario.")

if __name__ == "__main__":
    print("=== VERIFICADOR DE INTEGRIDAD DE IMÁGENES ===")
    print("Este script verificará todas las imágenes y reportará las corruptas.")
    print("NOTA: Este proceso puede tardar varios minutos dependiendo del tamaño del dataset.\n")
    
    # Verificar imágenes de entrenamiento
    print("Procesando conjunto de entrenamiento...")
    train_corrupted = verificar_imagenes(TRAIN_DIR)
    
    # Verificar imágenes de prueba
    print("\nProcesando conjunto de prueba...")
    test_corrupted = verificar_imagenes(TEST_DIR)
    
    # Combinar resultados
    todas_corruptas = train_corrupted + test_corrupted
    
    if todas_corruptas:
        print(f"\nSe encontraron un total de {len(todas_corruptas)} imágenes corruptas.")
        
        # Opcional: Guardar lista de imágenes corruptas en un archivo
        with open("imagenes_corruptas.txt", "w") as f:
            for img, error in todas_corruptas:
                f.write(f"{img}: {error}\n")
        print("Se ha guardado la lista en 'imagenes_corruptas.txt'")
        
        # Preguntar si se desean eliminar
        eliminar_imagenes_corruptas(todas_corruptas)
    else:
        print("\n✅ No se encontraron imágenes corruptas en ninguno de los conjuntos.")