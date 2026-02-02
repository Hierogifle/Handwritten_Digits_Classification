# 🎯 Classification de Chiffres Manuscrits MNIST

Une **application web interactive** de reconnaissance de chiffres manuscrits utilisant PyTorch. L'application offre trois interfaces pour soumettre un chiffre : upload d'image, dessin sur canvas ou capture photo en temps réel.

---

## 📋 Table des matières

- [Aperçu du projet](#-aperçu-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Modèles et Performance](#-modèles-et-performance)
- [Technologies](#-technologies)
- [Contribution](#-contribution)

---

## 🎨 Aperçu du projet

Ce projet implémente une **solution complète de reconnaissance de chiffres manuscrits** basée sur le dataset MNIST. L'application compare les prédictions de deux architectures de réseau de neurones :

- **MLP (Multi-Layer Perceptron)** : Réseau dense simple et rapide
- **CNN (Convolutional Neural Network)** : Réseau convolutif pour une meilleure extraction des caractéristiques

L'interface web permet de visualiser les prédictions des deux modèles simultanément avec leur niveau de confiance et la distribution de probabilité pour chaque chiffre.

---

## ✨ Fonctionnalités

### 🖼️ **Trois modes d'entrée**

1. **Upload d'image** : Téléchargez une image existante (PNG, JPG, GIF, BMP)
2. **Dessin interactif** : Dessinez un chiffre directement sur un canvas
3. **Capture photo** : Utilisez votre caméra pour photographier un chiffre

### 🔍 **Prédictions avancées**

- Prédictions simultanées avec **MLP et CNN**
- Scores de **confiance en pourcentage**
- Distribution de **probabilités** pour chaque chiffre (0-9)
- Temps de réponse **ultra-rapide** (< 100ms)

### 🎯 **Interface utilisateur**

- Design **responsive** et moderne
- Visualisation en temps réel des résultats
- Support **mobile, tablette et desktop**
- Stockage temporaire automatique des images

### ⚡ **Performance**

- Exécution GPU-compatible (CUDA)
- Fallback CPU automatique
- Nettoyage automatique des fichiers temporaires

---

## 🏗️ Architecture

### Modèle MLP

```
Entrée (784)
    ↓
[Couches denses + ReLU + Batch Norm + Dropout]
    ↓
Couches cachées (128, 64)
    ↓
Sortie (10)
```

**Caractéristiques :**
- 784 neurones d'entrée (28×28 pixels aplatis)
- Couches cachées configurables
- Activation ReLU avec Batch Normalization
- Dropout pour la régularisation
- Softmax sur la couche de sortie

### Modèle CNN

```
Entrée (1, 28, 28)
    ↓
[Conv2D → ReLU → BatchNorm → MaxPool] ×2
    ↓
[Conv2D → ReLU → BatchNorm → MaxPool]
    ↓
Couches denses (128)
    ↓
Sortie (10)
```

**Caractéristiques :**
- Filtres convolutifs progressifs (32, 64 canaux)
- Pooling maximal pour la réduction de dimension
- Batch Normalization après chaque couche
- Couches denses pour la classification finale

---

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip ou conda
- (Optionnel) CUDA 11.8+ pour accélération GPU

### 1. Cloner le repository

```bash
git clone https://github.com/DoreneABESSOLO/Handwritten-Digits-Classification.git
cd Handwritten-Digits-Classification
```

### 2. Créer un environnement virtuel

```bash
# Avec venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Ou avec conda
conda create -n mnist python=3.9
conda activate mnist
```

### 3. Installer les dépendances

```bash
pip install flask numpy pillow torch torchvision werkzeug
```

**Dépendances principales :**
- `Flask` - Framework web
- `PyTorch` - Framework de deep learning
- `NumPy` - Calculs numériques
- `Pillow` - Traitement d'images
- `torchvision` - Utilitaires PyTorch pour la vision

### 4. Télécharger les modèles pré-entraînés

Les modèles sont déjà présents dans le dossier `models/` :
- `model_final_MLP.pth` - Modèle MLP pré-entraîné
- `model_final_CNN.pth` - Modèle CNN pré-entraîné

---

## 💻 Utilisation

### Démarrer l'application

```bash
python app.py
```

L'application sera accessible à :
- **Local** : http://localhost:5000
- **Réseau** : http://{IP_locale}:5000 (ex: http://192.168.x.x:5000)

### Utilisation de l'interface

1. **Accédez à la page d'accueil** (`/`)
2. **Choisissez un mode d'entrée** :
   - `/upload` - Télécharger une image
   - `/draw` - Dessiner sur canvas
   - `/camera` - Prendre une photo
3. **Soumettez l'image** au serveur
4. **Visualisez les prédictions** avec les scores de confiance

### API REST

#### Endpoint de prédiction

```http
POST /api/predict
Content-Type: multipart/form-data

file: <image_file>
```

**Réponse** (exemple) :

```json
{
  "mlp_prediction": 7,
  "mlp_confidence": 99.45,
  "cnn_prediction": 7,
  "cnn_confidence": 99.89,
  "probabilities": {
    "mlp": [0.0, 0.05, 0.02, ..., 99.45, ...],
    "cnn": [0.01, 0.03, 0.01, ..., 99.89, ...]
  }
}
```

---

## 📁 Structure du projet

```
Handwritten-Digits-Classification/
│
├── app.py                          # Application Flask principale
│
├── templates/                      # Templates HTML
│   ├── index.html                 # Page d'accueil
│   ├── upload.html                # Page d'upload
│   ├── draw.html                  # Page de dessin
│   ├── camera.html                # Page caméra
│   └── result.html                # Page de résultats
│
├── static/                         # Fichiers statiques
│   ├── css/
│   │   └── style.css              # Styles CSS
│   └── temp/                      # Stockage temporaire des images
│
├── models/                         # Modèles pré-entraînés
│   ├── model_final_MLP.pth        # Checkpoint MLP
│   ├── model_final_CNN.pth        # Checkpoint CNN
│   └── parametre/
│       ├── mlp_best_hyperparams.json
│       └── cnn_best_hyperparams.json
│
├── scripts/                        # Notebooks Jupyter
│   ├── mlp.ipynb                  # Entraînement MLP
│   ├── cnn.ipynb                  # Entraînement CNN
│   └── EDA.ipynb                  # Analyse exploratoire
│
├── data/                           # Données CSV
│   ├── mnist_train.csv
│   └── mnist_test.csv
│
├── dataset/                        # Dataset MNIST brut
│   └── MNIST/raw/
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-labels-idx1-ubyte
│       ├── train-images-idx3-ubyte
│       └── train-labels-idx1-ubyte
│
├── documentation/                  # Documentation détaillée
│   ├── dataset_mnist.md           # Info sur MNIST
│   ├── mlp_info.md                # Détails MLP
│   ├── Perceptron_Multicouches.md # Théorie MLP
│   ├── data_analyse.md            # Analyse des données
│   └── to_do.md                   # Tâches en cours
│
|                     
│
├── README.md                       # Ce fichier
└── .gitignore                      # Fichiers ignorés
```

---

## 📊 Modèles et Performance

### Dataset MNIST

- **60 000** images d'entraînement
- **10 000** images de test
- Images en **niveaux de gris** 28×28 pixels
- Classes : chiffres 0-9

### Performance typique

| Modèle | Architecture | Taux d'erreur | Notes |
|--------|--------------|---------------|-------|
| MLP | 784→128→64→10 | ~2% | Rapide et léger |
| CNN | Conv+Pool+Dense | ~0.5-1% | Meilleure extraction de caractéristiques |
| État de l'art | SENet/Transformers | ~0.09% | Recherche avancée |

### Résultats de validation

Les hyperparamètres optimaux sont sauvegardés dans `models/parametre/` :

```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 50,
  "optimizer": "adam",
  "activation": "relu"
}
```

---

## 🛠️ Technologies

### Backend

| Technologie | Version | Usage |
|------------|---------|-------|
| Python | 3.8+ | Langage principal |
| Flask | 2.0+ | Framework web |
| PyTorch | 2.0+ | Deep learning |
| NumPy | 1.20+ | Calculs numériques |
| Pillow | 8.0+ | Traitement d'images |

### Frontend

- **HTML5** - Structure
- **CSS3** - Styling responsive
- **JavaScript** - Interactivité
- **Canvas API** - Dessin interactif
- **WebRTC API** - Accès caméra

### Déploiement

- GPU CUDA (optionnel)
- CPU compatible
- Navigateurs modernes (Chrome, Firefox, Safari, Edge)

---

## 📚 Notebooks Jupyter

### `scripts/mlp.ipynb`

Entraînement et évaluation du modèle MLP :
- Chargement du dataset MNIST
- Création de l'architecture MLP
- Boucles d'entraînement et validation
- Visualisation des résultats

### `scripts/cnn.ipynb`

Entraînement et évaluation du modèle CNN :
- Architecture convolutive
- Augmentation de données
- Courbes d'apprentissage
- Analyse de la performance

### `scripts/EDA.ipynb`

Analyse exploratoire des données :
- Visualisation du dataset
- Distribution des classes
- Exemples d'images
- Statistiques

---

## 🔧 Configuration avancée

### Utiliser le GPU

```python
# Dans app.py, le device est détecté automatiquement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation : {device}")
```

### Modifier les hyperparamètres

Éditez les fichiers JSON dans `models/parametre/` et modifiez l'architecture dans `app.py` :

```python
# Modifier la taille des couches cachées du MLP
layer_sizes = [256, 128, 64]  # Au lieu de [128, 64]
```

### Augmenter la limite de fichier

```python
# Dans app.py
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB au lieu de 16 MB
```

---

## 🐛 Dépannage

### L'application ne démarre pas

```bash
# Vérifier que Flask est installé
pip install flask

# Vérifier le port
python app.py --port 8000
```

### Erreur CUDA

```bash
# L'application détecte automatiquement le GPU
# Si vous ne voulez qu'utiliser CPU :
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Les modèles ne se chargent pas

```bash
# Vérifier que les fichiers .pth existent
ls models/model_final_*.pth

# Assurez-vous que PyTorch est installé
pip install torch
```

### Images temporaires non supprimées

Les images sont automatiquement nettoyées. Si problème :

```bash
# Vider manuellement
rm -rf static/temp/*  # Linux/Mac
rmdir /s static\temp\  # Windows
```

---

## 🎓 Ressources pédagogiques

### MNIST

- Dataset officiel : [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Kaggle : [MNIST Dataset](https://www.kaggle.com/hojjatk/mnist-dataset)

### PyTorch

- Documentation : [pytorch.org](https://pytorch.org)
- Tutoriels : [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Deep Learning

- Livre : "Deep Learning" - Goodfellow, Bengio, Courville
- Cours : [Stanford CS231n](http://cs231n.stanford.edu/)

---

**Dernière mise à jour** : 23 janvier 2026 | **Version** : 1.0