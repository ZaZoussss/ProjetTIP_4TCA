import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Charger la matrice
conf_matrix = np.load("conf_matrix.npy")
class_names = np.load("class_names.npy")

# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de Confusion")
plt.xticks(rotation=45)
plt.show()

# Calcul des prédictions correctes et totaux par classe
correct_per_class = np.diag(conf_matrix)  # Valeurs sur la diagonale
total_per_class = conf_matrix.sum(axis=1)  # Total par ligne (vrais labels)

# Calcul du taux de réussite
accuracy_per_class = correct_per_class / total_per_class

# Calcul du taux d'erreur (optionnel)
error_per_class = 1 - accuracy_per_class

# Tracer l’histogramme
plt.figure(figsize=(10, 6))
plt.bar(class_names, accuracy_per_class, color='green', label="Taux de réussite")
plt.bar(class_names, error_per_class, bottom=accuracy_per_class, color='red', label="Taux d'erreur")
plt.xlabel("Classes")
plt.ylabel("Proportion")
plt.title("Taux de réussite et d'erreur par classe")
plt.legend()
plt.xticks(rotation=45)
plt.show()