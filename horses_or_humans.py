import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np

def preprocess_image(image, size):
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(*size, 1)
    return image

def load_and_preprocess_data(dataset_name, image_size):
    dataset, metadata = tfds.load(dataset_name, as_supervised=True, with_info=True)
    train_data = []

    for i, (image, label) in enumerate(dataset["train"]):
        image = preprocess_image(image.numpy(), image_size)
        train_data.append([image, label])

    X, y = zip(*train_data)
    X = np.array(X).astype(float) / 255.0
    y = np.array(y)

    return X, y

def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model

def train_model(model, X, y, validation_split, epochs):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, validation_split=validation_split, epochs=epochs)

    return model

# Configuración
dataset_name = "horses_or_humans"
image_size = (100, 100)
validation_split = 0.15
epochs = 3

# Carga y preprocesamiento de datos
X, y = load_and_preprocess_data(dataset_name, image_size)

# Construcción del modelo
input_shape = (*image_size, 1)
model = build_model(input_shape)

# Entrenamiento del modelo
trained_model = train_model(model, X, y, validation_split, epochs)

# Guardar el modelo entrenado
trained_model.save("caballos_o_humanos.h5")