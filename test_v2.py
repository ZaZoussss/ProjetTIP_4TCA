import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

# Vérification de l'utilisation du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

# Hyperparamètres
img_size = (64, 64)
batch_size = 32
epochs = 5
learning_rate = 0.0005
num_classes = 4  # Nombre de classes (ajuste-le selon ton cas)

# Transforms pour la préparation des données
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Chargement des données
train_dir = "./resized_archive copie/Training Data/Training Data"
validation_dir = "./resized_archive copie/Validation Data/Validation Data"

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(validation_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Définition du modèle CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # Assuming 64x64 input
        self.fc2 = nn.Linear(128, num_classes)  # Multi-class output

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Pas de sigmoid ici, CrossEntropyLoss s'en occupe
        return x

# Initialisation du modèle
model = CNNModel().to(device)

# Définition du critère de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()  # Pour multi-class
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Fonction d'entraînement
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Pas besoin de view(-1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calcul de la précision
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = correct_preds / total_preds

        # Calcul de l'accuracy sur le set de validation
        val_accuracy, val_labels, val_preds = validate(model, val_loader)

        # Sauvegarder les pertes et les précisions
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_accuracies, val_labels, val_preds

# Fonction de validation
def validate(model, val_loader):
    model.eval()
    all_labels = []
    all_preds = []
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct_preds / total_preds
    return accuracy, np.array(all_labels), np.array(all_preds)

# Entraînement du modèle
train_losses, train_accuracies, val_accuracies, val_labels, val_preds = train(model, train_loader, criterion, optimizer, epochs)

# Matrice de confusion
cm = confusion_matrix(val_labels, val_preds)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()

# Affichage du rapport de classification
print("Classification Report:")
print(classification_report(val_labels, val_preds))

# Visualisation des courbes de perte et d'accuracy
plt.figure(figsize=(12, 6))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
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