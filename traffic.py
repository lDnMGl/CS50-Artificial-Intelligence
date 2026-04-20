import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    """
    images = []
    labels = []

    # Iteramos por cada categoría (carpetas del 0 al 42)
    for category in range(NUM_CATEGORIES):
        folder_path = os.path.join(data_dir, str(category))

        # Leemos cada imagen dentro de la carpeta
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)

            # Cargamos la imagen con OpenCV
            img = cv2.imread(img_path)
            if img is not None:
                # Redimensionamos a 30x30
                res_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(res_img)
                labels.append(category)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = tf.keras.models.Sequential([
        # 1. Capa Convolucional: Aprende 32 filtros de 3x3 para detectar bordes y texturas
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # 2. Max-Pooling: Reduce la resolución espacial manteniendo las características clave
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 3. Segunda Convolución: Detecta patrones más complejos (formas de señales)
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 4. Flatten: Aplana los mapas de características para pasar a las capas densas
        tf.keras.layers.Flatten(),

        # 5. Capa Oculta Densa: 128 neuronas para la toma de decisiones
        tf.keras.layers.Dense(128, activation="relu"),

        # 6. Dropout: Apaga el 50% de las neuronas aleatoriamente para evitar el sobreajuste
        tf.keras.layers.Dropout(0.5),

        # 7. Capa de Salida: 43 neuronas (una por categoría) con Softmax para obtener probabilidades
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compilamos usando Adam como optimizador y Cross-Entropy para pérdida multiclase
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
