"""
Application Flask de reconnaissance de chiffres manuscrits.

Cette application offre trois modes d'entrée pour soumettre un chiffre:
1. Upload d'une image existante
2. Dessin sur un canvas interactif
3. Capture photo via la caméra (mobile/desktop)

Les prédictions sont effectuées par deux modèles distincts (MLP et CNN) en PyTorch.
Toutes les images temporaires sont automatiquement supprimées.

Auteur: [Ton nom]
Date: 23 janvier 2026
Version: 1.0 - PyTorch
"""

from flask import Flask, render_template, request, jsonify, after_this_request
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import tempfile
import atexit
from datetime import datetime

# ============================================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================================

app = Flask(__name__)

# Dossier temporaire pour les images (sera nettoyé automatiquement)
TEMP_FOLDER = os.path.join('static', 'temp')

# Extensions de fichiers autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Configuration Flask
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite: 16MB

# Créer le dossier temporaire s'il n'existe pas
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Détection automatique du device (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DÉFINITION DES ARCHITECTURES DE MODÈLES
# ============================================================================

class MLP(nn.Module):
    """
    Architecture du Multi-Layer Perceptron pour MNIST.
    Version adaptée pour l'inférence Flask (sans transposition).
    
    Note : Pendant l'entraînement, tu utilisais le format (features, batch),
    mais en inférence avec Flask, on utilise le format standard (batch, features).
    """
    def __init__(self, layer_sizes, activation='relu', init_method='he',
                 dropout=0.0, batch_norm=False):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.use_batch_norm = batch_norm
        input_size = 784

        # Couches cachées
        for hidden_size in layer_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size

        # Couche de sortie
        self.output_layer = nn.Linear(input_size, 10)
        self.dropout = nn.Dropout(dropout)

        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()

        self._initialize_weights(init_method)

    def _initialize_weights(self, init_method):
        for layer in self.layers:
            if init_method == 'he':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif init_method == 'xavier':
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        if init_method == 'he':
            nn.init.kaiming_normal_(self.output_layer.weight)
        elif init_method == 'xavier':
            nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        """
        Forward pass pour l'inférence Flask.
        
        Args:
            x: Tensor de shape (batch_size, 1, 28, 28)
            
        Returns:
            Tensor de shape (batch_size, 10) contenant les logits
        """
        # Aplatir l'image : (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        
        # ✅ PAS de transposition - on est déjà au format (batch, features)
        # Pendant l'entraînement, tu avais (features, batch) d'où le x.T
        # Mais ici on part du format standard PyTorch
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        return x


class CNN(nn.Module):
    """
    Architecture du Convolutional Neural Network pour MNIST.
    
    Cette classe correspond EXACTEMENT à celle utilisée pendant l'entraînement.
    
    Hyperparamètres de ton modèle:
    - conv_channels: [64, 64]
    - fc_sizes: [160]
    - kernel_size: 5
    - dropout: 0.5
    - batch_norm: False
    - activation: gelu
    """
    def __init__(self, conv_channels=[32, 64], fc_sizes=[128],
                 kernel_size=3, pool_size=2, dropout=0.5,
                 batch_norm=True, activation='relu'):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_batch_norms = nn.ModuleList() if batch_norm else None
        self.use_batch_norm = batch_norm

        # Fonction d'activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()

        # Construire les couches convolutionnelles
        in_channels = 1  # MNIST = 1 canal (grayscale)
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernel_size, padding=1))
            if batch_norm:
                self.conv_batch_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.dropout_conv = nn.Dropout2d(dropout * 0.5)  # Dropout plus faible pour conv
        self.dropout_fc = nn.Dropout(dropout)

        # Calculer la taille dynamiquement avec un forward dummy
        flatten_size = self._get_flatten_size(conv_channels, kernel_size, pool_size)

        # Couches fully connected
        self.fc_layers = nn.ModuleList()
        self.fc_batch_norms = nn.ModuleList() if batch_norm else None

        in_features = flatten_size
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(in_features, fc_size))
            if batch_norm:
                self.fc_batch_norms.append(nn.BatchNorm1d(fc_size))
            in_features = fc_size

        # Couche de sortie
        self.output_layer = nn.Linear(in_features, 10)

        # Initialisation He pour ReLU
        self._initialize_weights()

    def _get_flatten_size(self, conv_channels, kernel_size, pool_size):
        """Calcule la taille du vecteur aplati avec un forward pass dummy"""
        with torch.no_grad():
            # Créer un tensor dummy (batch=1, channels=1, H=28, W=28)
            x = torch.zeros(1, 1, 28, 28)

            # Passer à travers les conv layers
            for i in range(len(conv_channels)):
                x = self.conv_layers[i](x)
                if self.use_batch_norm:
                    x = self.conv_batch_norms[i](x)
                x = self.activation(x)
                x = self.pool(x)

            # Calculer la taille après flatten
            return int(np.prod(x.size()[1:]))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass du CNN.
        
        Args:
            x: Tensor de shape (batch_size, 1, 28, 28)
            
        Returns:
            Tensor de shape (batch_size, 10) contenant les logits
        """
        # Couches convolutionnelles
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.conv_batch_norms[i](x)
            x = self.activation(x)
            x = self.pool(x)
            x = self.dropout_conv(x)

        # Aplatir
        x = x.view(x.size(0), -1)

        # Couches fully connected
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if self.use_batch_norm:
                x = self.fc_batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout_fc(x)

        # Sortie
        x = self.output_layer(x)
        return x

# ============================================================================
# CHARGEMENT DES MODÈLES PYTORCH
# ============================================================================

print("=" * 70)
print("Initialisation de l'application de reconnaissance de chiffres")
print("=" * 70)
print(f"Device utilisé: {device}")
print(f"Chargement des modèles PyTorch... [{datetime.now().strftime('%H:%M:%S')}]")

try:
    # Charger les dictionnaires contenant modèle + métadonnées
    mlp_checkpoint = torch.load('models/model_final_MLP.pth', 
                                map_location=device, 
                                weights_only=False)
    cnn_checkpoint = torch.load('models/model_final_CNN.pth', 
                                map_location=device, 
                                weights_only=False)
    
    # Extraire les hyperparamètres
    mlp_hyperparams = mlp_checkpoint.get('hyperparams', {})
    cnn_hyperparams = cnn_checkpoint.get('hyperparams', {})
    
    # Extraire les accuracies de validation
    mlp_val_acc = mlp_checkpoint.get('val_accuracy', 0)
    cnn_val_acc = cnn_checkpoint.get('val_accuracy', 0)
    
    # Instancier le MLP avec les hyperparamètres exacts
    mlp_model = MLP(
        layer_sizes=mlp_hyperparams.get('layer_sizes', [160, 112]),
        activation=mlp_hyperparams.get('activation', 'relu'),
        init_method=mlp_hyperparams.get('init_method', 'he'),
        dropout=mlp_hyperparams.get('dropout', 0.55),
        batch_norm=mlp_hyperparams.get('batch_norm', True)
    ).to(device)
    
    # Instancier le CNN avec les hyperparamètres exacts
    cnn_model = CNN(
        conv_channels=cnn_hyperparams.get('conv_channels', [64, 64]),
        fc_sizes=cnn_hyperparams.get('fc_sizes', [160]),
        kernel_size=cnn_hyperparams.get('kernel_size', 5),
        pool_size=2,  # Standard
        dropout=cnn_hyperparams.get('dropout', 0.5),
        batch_norm=cnn_hyperparams.get('batch_norm', False),
        activation=cnn_hyperparams.get('activation', 'gelu')
    ).to(device)
    
    # Charger les poids (state_dict) depuis les checkpoints
    mlp_model.load_state_dict(mlp_checkpoint['model_state_dict'])
    cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
    
    # Mettre les modèles en mode évaluation (IMPORTANT!)
    mlp_model.eval()
    cnn_model.eval()
    
    # Affichage des infos de chargement
    print("✓ MLP chargé avec succès")
    print(f"  - Architecture: {mlp_hyperparams.get('n_layers', len(mlp_hyperparams.get('layer_sizes', [])))} couches cachées")
    print(f"  - Neurones: {mlp_hyperparams.get('layer_sizes', [])}")
    print(f"  - Activation: {mlp_hyperparams.get('activation', '?').upper()}")
    print(f"  - Dropout: {mlp_hyperparams.get('dropout', '?')}")
    print(f"  - Batch Norm: {mlp_hyperparams.get('batch_norm', '?')}")
    print(f"  - Val Accuracy: {mlp_val_acc:.2%}")
    
    print("✓ CNN chargé avec succès")
    print(f"  - Conv layers: {cnn_hyperparams.get('n_conv_layers', len(cnn_hyperparams.get('conv_channels', [])))}")
    print(f"  - Channels: {cnn_hyperparams.get('conv_channels', [])}")
    print(f"  - Kernel size: {cnn_hyperparams.get('kernel_size', '?')}")
    print(f"  - FC sizes: {cnn_hyperparams.get('fc_sizes', [])}")
    print(f"  - Activation: {cnn_hyperparams.get('activation', '?').upper()}")
    print(f"  - Dropout: {cnn_hyperparams.get('dropout', '?')}")
    print(f"  - Batch Norm: {cnn_hyperparams.get('batch_norm', '?')}")
    print(f"  - Val Accuracy: {cnn_val_acc:.2%}")
    
    print(f"Modèles prêts ! [{datetime.now().strftime('%H:%M:%S')}]")
    print("=" * 70)
    
except FileNotFoundError as e:
    print(f"❌ ERREUR: Fichiers de modèles introuvables")
    print("Vérifiez que les fichiers suivants existent:")
    print("  - models/model_final_MLP.pth")
    print("  - models/model_final_CNN.pth")
    print(f"\nChemin actuel: {os.getcwd()}")
    exit(1)
    
except KeyError as e:
    print(f"❌ ERREUR: Clé manquante dans le checkpoint: {e}")
    print("Le fichier .pth ne contient pas 'model_state_dict'")
    exit(1)
    
except RuntimeError as e:
    print(f"❌ ERREUR lors du chargement des poids: {e}")
    print("\nL'architecture ne correspond pas aux poids sauvegardés.")
    print("Vérifiez les hyperparamètres dans le checkpoint:")
    print(f"MLP hyperparams: {mlp_checkpoint.get('hyperparams', {})}")
    print(f"CNN hyperparams: {cnn_checkpoint.get('hyperparams', {})}")
    exit(1)
    
except Exception as e:
    print(f"❌ ERREUR inattendue: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# NETTOYAGE AUTOMATIQUE DES FICHIERS TEMPORAIRES
# ============================================================================

def cleanup_temp_folder():
    """
    Nettoie tous les fichiers du dossier temporaire.
    
    Cette fonction est appelée automatiquement:
    - À l'arrêt de l'application (via atexit)
    - Après chaque requête qui génère un fichier temporaire
    
    Note:
        Les fichiers de plus de 1 heure sont également supprimés
        pour éviter l'accumulation en cas d'erreur.
    """
    try:
        current_time = datetime.now().timestamp()
        
        for filename in os.listdir(TEMP_FOLDER):
            filepath = os.path.join(TEMP_FOLDER, filename)
            
            # Supprimer les fichiers vieux de plus d'1 heure
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > 3600:  # 3600 secondes = 1 heure
                    os.remove(filepath)
                    print(f"🗑️  Fichier ancien supprimé: {filename}")
                    
    except Exception as e:
        print(f"Erreur lors du nettoyage: {e}")

# Enregistrer la fonction de nettoyage pour qu'elle s'exécute à l'arrêt de l'app
atexit.register(cleanup_temp_folder)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def allowed_file(filename):
    """
    Vérifie si le fichier a une extension autorisée.
    
    Args:
        filename (str): Nom du fichier à vérifier
        
    Returns:
        bool: True si l'extension est valide, False sinon
        
    Example:
        >>> allowed_file('digit.png')
        True
        >>> allowed_file('document.pdf')
        False
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Prétraite une image pour la reconnaissance de chiffres MNIST avec PyTorch.
    
    Pipeline de prétraitement:
    1. Chargement de l'image
    2. Conversion en niveaux de gris (grayscale)
    3. Redimensionnement à 28x28 pixels
    4. Inversion des couleurs si nécessaire (fond blanc -> fond noir)
    5. Normalisation des valeurs entre 0 et 1
    6. Conversion en tenseurs PyTorch
    
    Args:
        image_path (str): Chemin vers l'image à prétraiter
        
    Returns:
        tuple: Deux tenseurs PyTorch:
            - mlp_input (torch.Tensor): Shape (1, 1, 28, 28) pour le MLP
            - cnn_input (torch.Tensor): Shape (1, 1, 28, 28) pour le CNN
            
    Note:
        MNIST utilise des images avec fond noir et chiffre blanc.
        Si l'image a un fond blanc, elle est automatiquement inversée.
        Les deux modèles utilisent le même format d'entrée en PyTorch.
    """
    # Charger et convertir en niveaux de gris
    img = Image.open(image_path).convert('L')
    
    # Redimensionner à la taille MNIST standard
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convertir en array NumPy
    img_array = np.array(img)
    
    # Vérifier si l'image a un fond blanc (moyenne > 127)
    # MNIST utilise fond noir, donc on inverse si nécessaire
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normaliser les valeurs entre 0 et 1
    img_array = img_array / 255.0
    
    # Convertir en tenseur PyTorch
    # Shape: (1, 1, 28, 28) = (batch_size, channels, height, width)
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
    
    # Déplacer vers le device approprié (GPU ou CPU)
    img_tensor = img_tensor.to(device)
    
    # Pour PyTorch, MLP et CNN utilisent le même format d'entrée
    # Le MLP fera automatiquement le flatten dans son forward()
    return img_tensor, img_tensor


def make_predictions(mlp_input, cnn_input):
    """
    Effectue les prédictions avec les deux modèles PyTorch.
    """
    with torch.no_grad():
        mlp_logits = mlp_model(mlp_input)
        cnn_logits = cnn_model(cnn_input)
        
        mlp_probs = torch.softmax(mlp_logits, dim=1)
        cnn_probs = torch.softmax(cnn_logits, dim=1)
        
        # 🔍 DEBUG CNN
        print(f"🔍 CNN - Top 3 probabilités:")
        cnn_top3_probs, cnn_top3_indices = torch.topk(cnn_probs[0], 3)
        for i in range(3):
            print(f"   Classe {cnn_top3_indices[i].item()}: {cnn_top3_probs[i].item()*100:.4f}%")
        
        # 🔍 DEBUG MLP
        print(f"🔍 MLP - Top 3 probabilités:")
        mlp_top3_probs, mlp_top3_indices = torch.topk(mlp_probs[0], 3)
        for i in range(3):
            print(f"   Classe {mlp_top3_indices[i].item()}: {mlp_top3_probs[i].item()*100:.4f}%")
        
        mlp_probs_np = mlp_probs.cpu().numpy()[0]
        cnn_probs_np = cnn_probs.cpu().numpy()[0]
    
    results = {
        'mlp': {
            'prediction': int(np.argmax(mlp_probs_np)),
            'confidence': float(np.max(mlp_probs_np) * 100),
            'probabilities': [float(p) for p in mlp_probs_np]
        },
        'cnn': {
            'prediction': int(np.argmax(cnn_probs_np)),
            'confidence': float(np.max(cnn_probs_np) * 100),
            'probabilities': [float(p) for p in cnn_probs_np]
        }
    }
    
    return results


# ============================================================================
# ROUTES - PAGES PRINCIPALES
# ============================================================================

@app.route('/')
def index():
    """
    Page d'accueil de l'application.
    
    Affiche trois options pour soumettre un chiffre:
    - Upload d'image
    - Dessin sur canvas
    - Capture photo
    
    Returns:
        str: Template HTML de la page d'accueil
    """
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    """
    Page d'upload de fichier image.
    
    Permet à l'utilisateur de sélectionner une image depuis son appareil.
    
    Returns:
        str: Template HTML de la page d'upload
    """
    return render_template('upload.html')


@app.route('/draw')
def draw_page():
    """
    Page de dessin interactif.
    
    Affiche un canvas HTML5 où l'utilisateur peut dessiner un chiffre
    avec la souris (desktop) ou le doigt (mobile/tablette).
    
    Returns:
        str: Template HTML de la page de dessin
    """
    return render_template('draw.html')


@app.route('/camera')
def camera_page():
    """
    Page de capture photo via caméra.
    
    Utilise l'API getUserMedia pour accéder à la caméra de l'appareil
    et capturer une photo du chiffre.
    
    Returns:
        str: Template HTML de la page caméra
    """
    return render_template('camera.html')

# ============================================================================
# ROUTES - API DE PRÉDICTION
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Endpoint API pour effectuer une prédiction avec PyTorch.
    
    Cette route:
    1. Reçoit un fichier image (upload, canvas, ou caméra)
    2. Sauvegarde temporairement l'image
    3. Prétraite l'image
    4. Effectue les prédictions avec MLP et CNN (PyTorch)
    5. Supprime automatiquement l'image temporaire
    6. Retourne les résultats en JSON
    
    Request:
        POST avec multipart/form-data
        Champ 'file': Fichier image (PNG, JPG, etc.)
        
    Returns:
        JSON: {
            'mlp_prediction': int,
            'mlp_confidence': float,
            'cnn_prediction': int,
            'cnn_confidence': float,
            'probabilities': {
                'mlp': list,
                'cnn': list
            }
        }
        
    Errors:
        400: Fichier manquant ou type invalide
        500: Erreur lors du traitement
        
    Note:
        L'image est automatiquement supprimée après la réponse grâce
        au décorateur @after_this_request.
    """
    # Validation: vérifier la présence du fichier
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    
    # Validation: vérifier qu'un fichier est sélectionné
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Validation: vérifier l'extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Type de fichier non autorisé'}), 400
    
    try:
        # Générer un nom de fichier unique avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"temp_{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
        
        # Sauvegarder temporairement
        file.save(filepath)
        print(f"📥 Image reçue et sauvegardée: {filename}")
        
        # Prétraiter l'image
        mlp_input, cnn_input = preprocess_image(filepath)
        
        # Effectuer les prédictions
        results = make_predictions(mlp_input, cnn_input)
        print(f"🎯 Prédictions: MLP={results['mlp']['prediction']}, CNN={results['cnn']['prediction']}")
        
        # Décorateur pour supprimer le fichier APRÈS l'envoi de la réponse
        @after_this_request
        def remove_file(response):
            """
            Supprime le fichier temporaire après l'envoi de la réponse.
            
            Args:
                response: Objet Response Flask
                
            Returns:
                response: L'objet Response inchangé
            """
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"🗑️  Image temporaire supprimée: {filename}")
            except Exception as e:
                print(f"⚠️  Erreur lors de la suppression: {e}")
            return response
        
        # Formater la réponse JSON
        response_data = {
            'mlp_prediction': results['mlp']['prediction'],
            'mlp_confidence': round(results['mlp']['confidence'], 2),
            'cnn_prediction': results['cnn']['prediction'],
            'cnn_confidence': round(results['cnn']['confidence'], 2),
            'probabilities': {
                'mlp': [round(p * 100, 2) for p in results['mlp']['probabilities']],
                'cnn': [round(p * 100, 2) for p in results['cnn']['probabilities']]
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        
        # Nettoyer en cas d'erreur
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': f'Erreur lors du traitement: {str(e)}'}), 500

# ============================================================================
# GESTION D'ERREURS
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    """
    Gère l'erreur de fichier trop volumineux.
    
    Args:
        e: Exception Flask
        
    Returns:
        JSON: Message d'erreur
    """
    return jsonify({'error': 'Fichier trop volumineux (max 16MB)'}), 413


@app.errorhandler(500)
def internal_error(e):
    """
    Gère les erreurs internes du serveur.
    
    Args:
        e: Exception Flask
        
    Returns:
        JSON: Message d'erreur
    """
    return jsonify({'error': 'Erreur interne du serveur'}), 500

# ============================================================================
# POINT D'ENTRÉE DE L'APPLICATION
# ============================================================================

if __name__ == '__main__':
    """
    Lance le serveur Flask.
    
    Configuration:
        - debug=True: Mode développement avec rechargement automatique
        - host='0.0.0.0': Accessible depuis le réseau local (mobile)
        - port=5000: Port d'écoute
        
    Accès:
        - Local: http://localhost:5000
        - Réseau: http://[IP_de_ton_PC]:5000
    """
    print("\n🚀 Serveur Flask démarré !")
    print("📱 Accès local: http://localhost:5000")
    print("🌐 Accès réseau: http://10.10.98.111:5000")
    print("\nAppuyez sur Ctrl+C pour arrêter le serveur\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
