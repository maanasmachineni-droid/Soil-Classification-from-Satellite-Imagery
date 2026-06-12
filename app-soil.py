import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

MODEL_PATH = "models/soil_classifier.h5"

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model Loaded Successfully!")

# -------------------------------------------------
# CLASS LABELS
# MUST MATCH TRAINING ORDER
# -------------------------------------------------

categories = [

    "Alluvial_Soil",

    "Arid_Soil",

    "Black_Soil",

    "Laterite_Soil",

    "Mountain_Soil",

    "Red_Soil",

    "Yellow_Soil"
]

# -------------------------------------------------
# IMAGE SETTINGS
# -------------------------------------------------

IMG_SIZE = 128

# -------------------------------------------------
# TEST IMAGE
# -------------------------------------------------

IMAGE_PATH = "tesy_soil.jpg"

# -------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_COLOR
)

if image is None:

    print("Image not found.")

    exit()

# -------------------------------------------------
# PREPROCESS IMAGE
# -------------------------------------------------

image = cv2.resize(
    image,
    (IMG_SIZE, IMG_SIZE)
)

image = image.astype(np.float32)

image = image / 255.0

image = np.expand_dims(
    image,
    axis=0
)

# -------------------------------------------------
# PREDICT
# -------------------------------------------------

prediction = model.predict(image)

class_index = np.argmax(prediction)

confidence = np.max(prediction) * 100

# -------------------------------------------------
# RESULTS
# -------------------------------------------------

print("\nPrediction Results")

print("------------------")

print(
    f"Soil Type: "
    f"{categories[class_index]}"
)

print(
    f"Confidence: "
    f"{confidence:.2f}%"
)

# -------------------------------------------------
# ALL CLASS PROBABILITIES
# -------------------------------------------------

print("\nClass Probabilities")

print("-------------------")

for i in range(len(categories)):

    prob = prediction[0][i] * 100

    print(
        f"{categories[i]} : "
        f"{prob:.2f}%"
    )