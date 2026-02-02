# À Propos du Dataset

## Contexte

MNIST est un sous-ensemble d'un ensemble plus large disponible auprès du NIST (copié depuis [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) --> Plus disponible (10/12/2025)). 
Ces donénes sont également disponibles via [Kaggle](https://www.kaggle.com/hojjatk/mnist-dataset) et Keras :
```python
import keras.datasets.mnist as mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
et torchvision :
```python
import torchvision.datasets as datasets

# Charger avec torchvision (efficient)
transform = transforms.ToTensor()
train_data = datasets.MNIST(
    root=Path.cwd().parent / 'dataset', 
    train=True, 
    download=True, 
    transform=transform
)
```
## Description

Il s'agit d'un ensemble d'images de chiffres manuscrits, largement utilisé pour entraîner et tester des systèmes de traitement d'images et de reconnaissance de formes. Ces chiffres ont été extraits de formulaires remplis à la main par des employés du Bureau du recensement des États-Unis et des étudiants de l'Université de Virginie.

## Utilisation

De nombreuses méthodes ont été testées avec cet ensemble d'entraînement et cet ensemble de test (voir [http://yann.lecun.com/](http://yann.lecun.com/) pour plus de détails)

# Résultats typiques
| Année | Méthode                        | Type              | Taux d'erreur | Contribution clé
|-------|--------------------------------|-------------------|---------------|--------------------------------
| 1998  |  Classiﬁeur linéaire           | Non-RN            | 12%           | Baseline MNIST
| 2004  | LIRA (Neurones à 3 couches)    | RN simple         | 0.42%         | Première percée neuronale
| 2006  | SVM avec noyau polynomial      | Non-RN            | 0.56%         | Optimisation de marge
| 2011  | CNN 6 couches                  | RN profond        | 0.27%         | Architecture convolutionnelle
| 2013  | DropConnect                    | RN régularisé     | 0.21%         | Régularisation par désactivation
| 2016  | Committee de 35 CNN Ensemble   |                   | 0.17%         | Combinaison de modèles
| 2018  | SENet (Squeeze-and-Excitation) | RN attention      | 0.09%         | Mécanismes d'attention spatiale
| 2023  | Transformers convolutionnels   | Hybride CNN-Trans | 0.07%         | Fusion architectures

Record actuel : **0.18 % d’erreur**, détenu par le Département d'ingénierie des systèmes et de l'information de l’Université
de Virginie.

## Comment lire les données

Voir [exemple de lecteur MNIST](https://www.kaggle.com/hojjatk/read-mnist-dataset) ou `minst_database.ipynb` pour un exemple de code Python permettant de lire les fichiers MNIST.

## Contenu générale

La base de données MNIST de chiffres manuscrits contient un ensemble d'entraînement de 60 000 exemples et un ensemble de test de 10 000 exemples. Quatre fichiers sont disponibles :

- **train-images-idx3-ubyte.gz** : images de l'ensemble d'entraînement (9 912 422 octets)
- **train-labels-idx1-ubyte.gz** : étiquettes de l'ensemble d'entraînement (28 881 octets)
- **t10k-images-idx3-ubyte.gz** : images de l'ensemble de test (1 648 877 octets)
- **t10k-labels-idx1-ubyte.gz** : étiquettes de l'ensemble de test (4 542 octets)