import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# Définition du modèle CNN (extrait de votre script)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 15)  # Ajusté au nombre de classes
        self.dropout = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(torch.relu(self.batch_norm4(self.conv4(x))))
        x = self.pool(torch.relu(self.batch_norm5(self.conv5(x))))
        x = x.view(-1, 512 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Charger le modèle
def load_model(filename="model.pth"):
    device = torch.device("cpu")  # Utilisation du CPU par défaut
    model = CNNModel().to(device)
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modèle chargé à partir de {filename}")
    return model

# Préparation des données de test
def prepare_test_data(test_dir, img_size=(64, 64), batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader, test_data.classes

# Évaluation sur le jeu de test
def evaluate_model(model, test_loader, class_names):
    device = torch.device("cpu")  # Assurer que l'évaluation est sur le CPU
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Rapport de classification
    print("Rapport de classification :")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités')
    plt.title('Matrice de confusion')
    plt.show()

# Histogramme des taux de bonne classification par classe
def plot_class_accuracy(model, test_loader, class_names):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    correct_per_class = np.zeros(len(class_names), dtype=int)
    total_per_class = np.zeros(len(class_names), dtype=int)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for label, pred in zip(labels, preds):
                total_per_class[label.item()] += 1
                if label == pred:
                    correct_per_class[label.item()] += 1

    # Calcul des taux de bonne classification
    class_accuracy = correct_per_class / total_per_class

    # Affichage de l'histogramme
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_names)), class_accuracy, color='skyblue')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.xlabel("Classes")
    plt.ylabel("Taux de bonne classification")
    plt.title("Taux de bonne classification par classe")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    test_dir = "./resized_archive/Testing Data/Testing Data"  # Remplacez par le chemin vers vos données de test
    model_path = "model.pth"  # Chemin vers le fichier du modèle

    # Charger le modèle
    model = load_model(model_path)

    # Préparer les données de test
    test_loader, class_names = prepare_test_data(test_dir)

    # Évaluer le modèle
    evaluate_model(model, test_loader, class_names)

    # Afficher le taux de bonne classification par classe
    plot_class_accuracy(model, test_loader, class_names)