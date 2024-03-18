# Funciones auxiliares

import tensorflow as tf
import matplotlib.pyplot as plt


# No usar la GPU
def disable_gpu():
    tf.config.set_visible_devices([], 'GPU')


# Reservar memoria a medida que se necesite
# Si se usa, debe llamarse antes de inicializar las GPUs
def gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:  # Debe ser igual para cada GPU
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}\nLogical GPUs: {len(logical_gpus)}")
    except RuntimeError as error:
        print(error)


# Gráfico del resultado del entrenamiento
def plot_history(history, metrics, names):
    for metric, name in zip(metrics, names):
        plt.plot(history[metric], label=name)
        plt.legend(loc='lower right')

    plt.show()
