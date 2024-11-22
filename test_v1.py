import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

import pathlib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Set your images directories
train_data_dir = pathlib.Path("./resized_archive/Training Data/Training Data")
validation_data_dir = pathlib.Path("./resized_archive/Validation Data/Validation Data")
# List all image files with .jpg, .jpeg, or .png extensions
train_image_files = list(train_data_dir.glob('*/*.jpg')) + list(train_data_dir.glob('*/*.jpeg')) + list(
    train_data_dir.glob('*/*.png'))
train_image_count = len(train_image_files)
print(f"Found {train_image_count} images for training")
images_height = 64
images_width = 64
# cats = list(data_dir.glob('Cat/*'))
# PIL.Image.open(str(cats[0]))
# print(cats[0])
batch_size = 300
v_batch_size = 100
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
    batch_size=v_batch_size,
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
train_ds = train_ds.shuffle(batch_size * 2).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
# Creating the model
num_classes = len(class_names)  # number of classes
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(images_height, images_width, 3)),
    # changing RGB values from [0, 255] to [0, 1]

    layers.Conv2D(8, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(8, 5, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(12, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    # Transformer la sortie 4D en 3D
    layers.Flatten(),
    # layers.Reshape((-1, 16)),
    # layers.Dense(64),
    # layers.Softmax(),
    # layers.SimpleRNN(units=100, recurrent_initializer='RandomNormal'),
    # layers.Softmax(),
    # # layers.SimpleRNN(units=10),
    #
    #
    # layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
# Creating opimizer
# optimize = Adam(learning_rate=0.001)
optimize = SGD(learning_rate=0.001)
# Compiling model
model.compile(optimizer=optimize,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
# Model summary
# model.summary() # displays the summary information about the created model
# Model training
epochs = 20
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

# Récupération des vraies étiquettes et des prédictions
true_labels = np.concatenate([y for x, y in validation_ds], axis=0)  # Étiquettes réelles
predicted_labels = np.argmax(model.predict(validation_ds), axis=1)  # Prédictions

# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de Confusion")
plt.xticks(rotation=45)
plt.show()