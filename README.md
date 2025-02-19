import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Ensure dataset path is correctly formatted
DATASET_PATH = "/kaggle/input/dataset"  # Update this to your dataset path
if not isinstance(DATASET_PATH, str):
    raise ValueError("DATASET_PATH must be a string containing the dataset directory path")

EPOCHS = 10

# Load and preprocess dataset
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained model (EfficientNetB0) and modify it
try:
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3), 
        include_top=False, 
        weights='imagenet'  # Change to None if no internet access
    )
except Exception as e:
    print("Could not download weights, initializing model without pre-trained weights.")
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3), 
        include_top=False, 
        weights=None  # Train from scratch
    )

base_model.trainable = False  # Freeze the base model

# Create a new classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Fine-tune by unfreezing some layers
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save the fine-tuned model
model.save("fine_tuned_efficientnet.h5")
print("Model training complete and saved!")
