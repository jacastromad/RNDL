import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import layers
from util import gpu_growth, plot_history


CLASSES = 10                # Número de clases
INPUT_SHAPE = (28, 28, 1)   # Dimensiones de los ejemplos
BATCH_SIZE = 128            # Tamaño del lote para entrenamiento
EPOCHS = 15                 # Épocas de entrenamiento


gpu_growth()

# Cargar los conjuntos de ejemplos de entrenamiento y test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Si ya tenemos los datos descargados y el fichero es 'mnist.npz':
# with np.load('mnist.npz', allow_pickle=True) as data:
#     x_train, y_train = data['x_train'], data['y_train']
#     x_test, y_test = data['x_test'], data['y_test']

# Normalizar
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Añadir una dimensión para tener tensores (28, 28, 1) en vez de (28, 28)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("Conjunto de entrenamiento:", x_train.shape[0], "elementos.")
print("Conjunto de test:", x_test.shape[0], "elementos.")
print("Dimensiones conjunto entrenamiento:", x_train.shape)

# Mostrar unos ejemplos
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(x_train[i], cmap="gray")
plt.show()

input("Enter para contiuar...")

# Convertir a las dimensiones de la salida esperada
# 5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
y_train = keras.utils.to_categorical(y_train, CLASSES)
y_test = keras.utils.to_categorical(y_test, CLASSES)

# Modelo con capas totalmente conectadas
model = keras.Sequential(
    [
        keras.Input(shape=INPUT_SHAPE),
        layers.Flatten(),
        layers.Dense(64, activation="sigmoid"),
        layers.Dense(32, activation="sigmoid"),
        layers.Dense(CLASSES, activation="softmax"),
    ]
)

# Modelo con capas convolucionales
# model = keras.Sequential(
#     [
#         keras.Input(shape=INPUT_SHAPE),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.4),
#         layers.Dense(CLASSES, activation="softmax"),
#     ]
# )

# Mostrar resumen del modelo
model.summary()
input("Enter para contiuar...")

# Seleccionar función coste (loss), optimizador y métricas
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# Entrenar el modelo utilizando el 10% de los ejemplos para validación
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.1)

# Evaluación del modelo con el conjunto de ejemplos de test
eval = model.evaluate(x_test, y_test, verbose=0)
print("Coste test:", eval[0])
print("Exactitud test:", eval[1])

plot_history(history.history, ["accuracy", "val_accuracy"], ["acc", "val acc"])
input("Enter para continuar...")

# Predicciones del modelo para todo el conjunto de test
pred_test = model.predict(x_test)
pred_test_class = np.argmax(pred_test, axis=-1)
y_test_class = np.argmax(y_test, axis=-1)

# Dígitos mal clasificados
errors = np.where(pred_test_class != y_test_class)[0]

# Mostrar los 6 primeros dígitos mal clasificados
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(x_test[errors[i]], cmap="gray")
    predicted = pred_test_class[errors[i]]
    expected = y_test_class[errors[i]]
    ax.set_title("Pred: "+str(predicted)+" - Exp: "+str(expected))
plt.show()
