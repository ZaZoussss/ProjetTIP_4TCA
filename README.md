# Projet TIP
## DeepLearning - 4TCA INSA Lyon
Le module TIP "Traitement de l'Image et de la Parole" faisant suite au module MAS "Mathématiques Appliqués au Signal" permet de découvrir tous les différentes transformations appliquées aux images et leurs concepts mathématiques associés. Ce module introduit également les réseaux neuronaux et leur utilisation dans le traitement des images, et notamment la classification.

Après avoir travaillé en TP sur les bases CIFAR10 et MNIST, un projet a été lancé pour nous évaluer sur nos aptitudes à effectuer du traitement d'images. Notre groupe composé de [ZaZoussss](https://github.com/ZaZoussss), [Thomas G](https://github.com/TomasGith) et [F4JOV](https://github.com/F4JOV) a choisit de travailler sur de la classification d'images en entraînant un modèle d'apprentissage. Après avoir consulté plusieurs projets sur [Kaggle](www.kaggle.com), nous avons trouvé ce [dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset) contenant 30000 images d'animaux répartis en 15 classes.

Nous avons ensuite construit et entraîné notre propre modèle en utilisant Tensorflow puis PyTorch. Ce reopository contient plusieurs fichiers et dossiers :
- ```Final_script.py``` : Ce fichier contient le code du modèle non entraîné et prêt à l'être en y appliquant un dataset.
- ```exported_model\``` : Ce dossier contient le modèle exporté ainsi qu'un programme Python qui l'importe, lui passe un dataset et affiche les résultats (loss, accuracy, confusion matrix).
- ```images\``` : Ce dossier contient les images de la matrice de confusion, des courbes de précision et de perte, des connexions entre couches de notre réseau, ainsi que le programme permettant de générer ces connexions.

- ```script_resize\``` : Ce dossier contient un script permettant de modifier la taille des images de notre dataset de 256×256 en 64×64.

- ```script_tests\``` : Ce dossier contient certains de nos programmes sur lesquels nous avons travaillé afin de tester nos modèles avant de trouver celui que nous retenons pour le projet.