import os

# -------------------------------------------------
# Reduce TensorFlow warnings
# -------------------------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# -------------------------------------------------
# GPU MEMORY CONTROL
# Prevent TensorFlow from consuming all VRAM
# -------------------------------------------------

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        print("GPU Enabled Successfully")

    except RuntimeError as e:

        print(e)

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

IMG_SIZE = 128

BATCH_SIZE = 8

EPOCHS = 15

DATASET_PATH = "dataset_png"

# -------------------------------------------------
# VERIFY DATASET
# -------------------------------------------------

print("\nDataset Summary")
print("----------------")

for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(
        DATASET_PATH,
        category
    )

    if os.path.isdir(category_path):

        count = len(os.listdir(category_path))

        print(f"{category} → {count} images")

# -------------------------------------------------
# DATA AUGMENTATION
# -------------------------------------------------

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.2
)

# -------------------------------------------------
# TRAINING DATA
# -------------------------------------------------

train_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='sparse',

    subset='training'
)

# -------------------------------------------------
# VALIDATION DATA
# -------------------------------------------------

val_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='sparse',

    subset='validation'
)

# -------------------------------------------------
# PRINT CLASS INDICES
# IMPORTANT FOR app.py
# -------------------------------------------------

print("\nClass Indices")
print("-------------")

print(train_generator.class_indices)

# -------------------------------------------------
# NUMBER OF CLASSES
# -------------------------------------------------

NUM_CLASSES = len(
    train_generator.class_indices
)

print(f"\nTotal Classes: {NUM_CLASSES}")

# -------------------------------------------------
# CNN MODEL
# -------------------------------------------------

model = Sequential([

    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        NUM_CLASSES,
        activation='softmax'
    )
])

# -------------------------------------------------
# COMPILE MODEL
# Lower learning rate prevents NaN loss
# -------------------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)

# -------------------------------------------------
# MODEL SUMMARY
# -------------------------------------------------

print("\nModel Summary")
print("-------------")

model.summary()

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

print("\nStarting Training...")

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS
)

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/land_classifier.h5"

model.save(MODEL_PATH)

print("\nModel Saved Successfully!")
print(f"Saved at: {MODEL_PATH}")

# -------------------------------------------------
# FINAL ACCURACY
# -------------------------------------------------

final_train_acc = history.history['accuracy'][-1]

final_val_acc = history.history['val_accuracy'][-1]

print("\nTraining Complete")
print("-----------------")

print(f"Final Training Accuracy: {final_train_acc:.4f}")

print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# -------------------------------------------------
# OPTIONAL TRAINING GRAPH
# -------------------------------------------------

try:

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history['accuracy']
    )

    plt.plot(
        history.history['val_accuracy']
    )

    plt.title('Model Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend([
        'Train',
        'Validation'
    ])

    plt.show()

except:

    print("Matplotlib not installed. Skipping graph.")

