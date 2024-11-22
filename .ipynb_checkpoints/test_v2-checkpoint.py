import tensorflow as tf
from tensorflow.keras import layers, Sequential
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Définir les chemins des données
train_data_dir = pathlib.Path("./resized_archive/Training Data/Training Data")
validation_data_dir = pathlib.Path("./resized_archive/Validation Data/Validation Data")

# Dimensions des images
images_height = 64
images_width = 64
batch_size = 64

# Chargement des ensembles de données avec augmentation pour l'entraînement
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    image_size=(images_height, images_width),
    batch_size=batch_size,
    shuffle=True
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_data_dir,
    image_size=(images_height, images_width),
    batch_size=batch_size
)

# Récupération des noms des classes
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes détectées : {class_names}")

# Prétraitement des données pour performances
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Augmentation des données
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.2, 0.2),
])

# Définition du modèle CNN
model = Sequential([
    layers.InputLayer(input_shape=(images_height, images_width, 3)),  # Définition explicite de la forme d'entrée
    data_augmentation,  # Augmentation appliquée en temps réel
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.7),  # Dropout modéré pour plus d'apprentissage
    layers.Dense(num_classes, activation='softmax')  # Classification en 15 classes
])

# Compilation du modèle avec un taux d'apprentissage plus bas
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Taux d'apprentissage plus bas pour plus de stabilité

# Callback ReduceLROnPlateau pour ajuster le taux d'apprentissage
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Ajout d'EarlyStopping pour stopper l'entraînement en cas de stagnation
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Entraînement avec callbacks
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=50,
    callbacks=[reduce_lr, early_stopping]
)

# Affichage des résultats
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()