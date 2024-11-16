import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import pathlib

# Set your images directories
train_data_dir = pathlib.Path(r"L:\big_file_storage\4TCA_S1_TIP\animals_15_classes\resized_archive\Training Data\Training Data")
validation_data_dir = pathlib.Path(r"L:\big_file_storage\4TCA_S1_TIP\animals_15_classes\resized_archive\Validation Data\Validation Data")
# train_data_dir = pathlib.Path("./resized_archive_light/Training Data/Training Data")
# validation_data_dir = pathlib.Path("./resized_archive_light/Validation Data/Validation Data")

# List all image files with .jpg, .jpeg, or .png extensions
train_image_files = list(train_data_dir.glob('*/*.jpg')) + list(train_data_dir.glob('*/*.jpeg')) + list(train_data_dir.glob('*/*.png'))
train_image_count = len(train_image_files)

print(f"Found {train_image_count} images for training")

images_height = 64
images_width = 64

# cats = list(data_dir.glob('Cat/*'))
# PIL.Image.open(str(cats[0]))
# print(cats[0])

batch_size = 32

# Creating training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    color_mode='rgb',
    # batch_size=train_image_count,
    batch_size=batch_size,
    image_size=(images_height, images_width),
    shuffle=True,
    seed=None,
    follow_links=False,
    verbose=True
)

# Creating validation dataset
validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_data_dir,
    color_mode='rgb',
    # batch_size=train_image_count,
    batch_size=batch_size,
    image_size=(images_height, images_width),
    shuffle=True,
    seed=None,
    follow_links=False,
    verbose=True
)

# Detected class names
class_names = train_ds.class_names
print(class_names)

# Configuring dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

# train_ds = train_ds.batch(batch_size).cache(r"L:\big_file_storage\4TCA_S1_TIP\cache\train.cache").shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# validation_ds = validation_ds.batch(batch_size).cache(r"L:\big_file_storage\4TCA_S1_TIP\cache\valid.cache").prefetch(buffer_size=AUTOTUNE)

# Creating the model
num_classes = len(class_names) # number of classes
model = Sequential([
  layers.Rescaling(1./255, input_shape=(images_height, images_width, 3)), # changing RGB values from [0, 255] to [0, 1]
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compiling model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model summary
# model.summary() # displays the summary information about the created model


# Model training
epochs=10
history = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=epochs
)

# Results display
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()