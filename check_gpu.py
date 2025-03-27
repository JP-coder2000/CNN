import tensorflow as tf
import os

# Verificar versión de TensorFlow y soporte de GPU (Metal)
print(f"TensorFlow version: {tf.__version__}")
print(f"Metal GPU support: {'Yes' if tf.config.list_physical_devices('GPU') else 'No'}")

# Listar dispositivos disponibles
devices = tf.config.list_physical_devices()
print("\nDispositivos detectados:")
for device in devices:
    print(f"- {device}")

# Ejecutar operación de prueba en GPU
try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([2000, 2000])
        b = tf.random.normal([2000, 2000])
        
        print("\nRealizando operación de prueba en GPU...")
        c = tf.matmul(a, b)
        
    print("\n✅ Operación completada con GPU Metal!")
    print(f"Resultado shape: {c.shape}")
    print(f"Dispositivo usado: {c.device}")

except Exception as e:
    print("\n❌ Error al usar GPU:", e)