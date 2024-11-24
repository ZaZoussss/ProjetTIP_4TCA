import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import time  # Importation du module time

# Vérification de l'utilisation du GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

# Hyperparamètres
img_size = (64, 64)
batch_size = 32
epochs = 50
learning_rate = 0.0005
num_classes = 15  # Nombre de classes (ajuste-le selon ton cas)

train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomRotation(30),          # Rotation aléatoire entre -30° et 30°
    transforms.RandomHorizontalFlip(),      # Flip horizontal
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), shear=10),  # Ajout de degrees pour RandomAffine
    transforms.RandomResizedCrop(img_size[0], scale=(0.8, 1.0)),  # Découpe aléatoire
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Variation des couleurs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformation de validation sans augmentation
val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Chargement des données
train_dir = "./resized_archive/Training Data/Training Data"
validation_dir = "./resized_archive/Validation Data/Validation Data"

train_data = datasets.ImageFolder(train_dir, transform=train_transform)  # Application des augmentations
val_data = datasets.ImageFolder(validation_dir, transform=val_transform)  # Pas d'augmentation pour la validation

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Définition du modèle CNN avec plus de couches convolutives et Dropout
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Nouvelle couche convolutive
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Nouvelle couche convolutive
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 256)  # Adapté à l'augmentation du nombre de canaux
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.1)  # Augmentation du taux de dropout
        self.batch_norm1 = nn.BatchNorm2d(32)  # Batch Normalization après chaque conv1
        self.batch_norm2 = nn.BatchNorm2d(64)  # Batch Normalization après chaque conv2
        self.batch_norm3 = nn.BatchNorm2d(128)  # Batch Normalization après chaque conv3
        self.batch_norm4 = nn.BatchNorm2d(256)  # Batch Normalization après chaque conv4
        self.batch_norm5 = nn.BatchNorm2d(512)  # Batch Normalization après chaque conv5

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))  # BatchNorm après conv1
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))  # BatchNorm après conv2
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))  # BatchNorm après conv3
        x = self.pool(torch.relu(self.batch_norm4(self.conv4(x))))  # BatchNorm après conv4
        x = self.pool(torch.relu(self.batch_norm5(self.conv5(x))))  # BatchNorm après conv5
        x = x.view(-1, 512 * 2 * 2)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialisation du modèle
model = CNNModel().to(device)  # Assurer que le modèle est sur le même device que le GPU

# Définition du critère de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()  # Pour multi-class
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialisation du scheduler (Reduit le taux d'apprentissage si la validation n'améliore pas)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

# Fonction d'entraînement
def train(model, train_loader, criterion, optimizer, scheduler, epochs):
    model.train()
    train_losses = []
    val_losses = []  # Tableau pour la perte de validation
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()

        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calcul de la précision
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = correct_preds / total_preds

        # Calcul de la précision et de la perte sur le set de validation
        val_accuracy, val_labels, val_preds, val_loss = validate(model, val_loader, criterion)

        # Sauvegarder les pertes et les précisions
        train_losses.append(avg_loss)
        val_losses.append(val_loss)  # Enregistrer la perte de validation
        train_accuracies.append(avg_accuracy)
        val_accuracies.append(val_accuracy)

        # Mettre à jour le scheduler
        scheduler.step(val_accuracy)

        # Calcul du temps écoulé et affichage
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds")

    return train_losses, val_losses, train_accuracies, val_accuracies, val_labels, val_preds

# Fonction de validation
def validate(model, val_loader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    correct_preds = 0
    total_preds = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct_preds / total_preds
    avg_loss = running_loss / len(val_loader)  # Calcul de la perte moyenne
    return accuracy, np.array(all_labels), np.array(all_preds), avg_loss

# Entraînement du modèle
train_losses, val_losses, train_accuracies, val_accuracies, val_labels, val_preds = train(model, train_loader, criterion, optimizer, scheduler, epochs)

# Sauvegarde du modèle
def save_model(model, optimizer, epoch, filename="./model.pth"):
    # Sauvegarder l'état du modèle et de l'optimiseur
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_losses[-1],  # Dernière valeur de perte de validation
    }, filename)
    print(f"Modèle sauvegardé sous {filename}")

# Exemple d'appel à la fonction après l'entraînement
save_model(model, optimizer, epochs)

# Matrice de confusion
cm = confusion_matrix(val_labels, val_preds)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()

# Rapport de classification
print("Classification Report:")
print(classification_report(val_labels, val_preds))

# Visualisation des courbes
# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')  # Ajouter la courbe de perte de validation
plt.title('Courbe de perte')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()

# Courbe de précision (train et validation)
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Courbe de précision')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()

plt.show()


# Rechargement du modèle
# def load_model(model, optimizer, filename="model.pth"):
#    checkpoint = torch.load(filename)
#    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#    epoch = checkpoint['epoch']
#    loss = checkpoint['loss']
#    print(f"Modèle chargé à partir de {filename}, époque {epoch}, perte {loss:.4f}")
#    return model, optimizer, epoch, loss

# Exemple d'appel pour charger le modèle
# model, optimizer, epoch, loss = load_model(model, optimizer)