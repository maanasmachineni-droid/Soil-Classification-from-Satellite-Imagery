import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

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
# -------------------------------------------------

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        print("GPU Enabled")

    except RuntimeError as e:

        print(e)

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

IMG_SIZE = 128

BATCH_SIZE = 8

EPOCHS = 15

DATASET_PATH = "dataset_soil"

# -------------------------------------------------
# DATA AUGMENTATION
# -------------------------------------------------

datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.2
)

# -------------------------------------------------
# TRAINING DATA
# -------------------------------------------------

train_generator = datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='sparse',

    subset='training'
)

# -------------------------------------------------
# VALIDATION DATA
# -------------------------------------------------

val_generator = datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='sparse',

    subset='validation'
)

# -------------------------------------------------
# PRINT CLASS INDICES
# -------------------------------------------------

print("\nClass Indices")

print(train_generator.class_indices)

# -------------------------------------------------
# NUMBER OF CLASSES
# -------------------------------------------------

NUM_CLASSES = len(
    train_generator.class_indices
)

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

model.summary()

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS
)

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

os.makedirs("models", exist_ok=True)

model.save(
    "models/soil_classifier.h5"
)

print("\nModel Saved Successfully!")