"""
utils.py - Fonctions utilitaires communes pour le projet MLOps

Ce module contient les fonctions partagées entre les différents scripts :
- Préprocessing d'images
- Chargement/sauvegarde de modèles
- Configuration ClearML
- Gestion des chemins
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


# =============================================================================
# CONFIGURATION ET CHEMINS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEEDBACK_DIR = DATA_DIR / "feedback_labeled"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

# Configuration ClearML
CLEARML_PROJECT_NAME = "MLOps_CatsVsDogs"
CLEARML_DATASET_BASE_NAME = "cats_vs_dogs_base"
CLEARML_DATASET_FEEDBACK_NAME = "cats_vs_dogs_feedback"

# Classes pour Cats vs Dogs
DEFAULT_CLASSES = ["cat", "dog"]

# Paramètres d'entraînement par défaut
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10
DEFAULT_IMAGE_SIZE = (224, 224)


# =============================================================================
# PRÉPROCESSING D'IMAGES
# =============================================================================

def get_train_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Retourne les transformations pour l'entraînement avec augmentation.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Retourne les transformations pour l'inférence (sans augmentation).
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(
    image: Image.Image,
    for_training: bool = False
) -> torch.Tensor:
    """
    Préprocesse une image PIL pour le modèle.
    
    Args:
        image: Image PIL à préprocesser
        for_training: Si True, applique l'augmentation de données
        
    Returns:
        Tensor préprocessé
    """
    if for_training:
        transform = get_train_transforms()
    else:
        transform = get_inference_transforms()
    
    # Convertir en RGB si nécessaire
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return transform(image)


# =============================================================================
# MODÈLE
# =============================================================================

def create_model(
    num_classes: int = 5,
    pretrained: bool = True,
    model_name: str = "resnet18"
) -> nn.Module:
    """
    Crée un modèle de classification d'images.
    
    Args:
        num_classes: Nombre de classes de sortie
        pretrained: Utiliser les poids pré-entraînés
        model_name: Nom du modèle ("resnet18", "resnet34", "resnet50")
        
    Returns:
        Modèle PyTorch
    """
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
    # Remplacer la dernière couche pour notre nombre de classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def save_model(
    model: nn.Module,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Sauvegarde le modèle et ses métadonnées.
    
    Args:
        model: Modèle PyTorch à sauvegarder
        path: Chemin de sauvegarde
        metadata: Métadonnées optionnelles (accuracy, epoch, etc.)
        
    Returns:
        Chemin du fichier sauvegardé
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), path)
    
    # Sauvegarder les métadonnées si fournies
    if metadata:
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return path


def load_model(
    path: Path,
    num_classes: int = None,
    model_name: str = "resnet18",
    device: str = "cpu"
) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Charge un modèle sauvegardé et ses métadonnées.
    
    Args:
        path: Chemin du fichier modèle
        num_classes: Nombre de classes (auto-détecté depuis metadata si None)
        model_name: Architecture du modèle
        device: Device ('cpu' ou 'cuda')
        
    Returns:
        Tuple (modèle, métadonnées)
    """
    path = Path(path)
    
    # D'abord charger les métadonnées pour obtenir num_classes
    metadata = None
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    
    # Déterminer num_classes depuis les métadonnées ou le défaut
    if num_classes is None:
        if metadata and "classes" in metadata:
            num_classes = len(metadata["classes"])
            print(f"Classes détectées depuis metadata: {metadata['classes']}")
        elif metadata and "num_classes" in metadata:
            num_classes = metadata["num_classes"]
        else:
            # Défaut pour Cats vs Dogs
            num_classes = len(DEFAULT_CLASSES)
            print(f"Utilisation du défaut: {num_classes} classes ({DEFAULT_CLASSES})")
    
    # Récupérer l'architecture depuis les métadonnées si disponible
    if metadata and "model_architecture" in metadata:
        model_name = metadata["model_architecture"]
    
    # Créer le modèle et charger les poids
    model = create_model(num_classes=num_classes, pretrained=False, model_name=model_name)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, metadata


def get_latest_model_path(models_dir: Path = MODELS_DIR) -> Optional[Path]:
    """
    Récupère le chemin du modèle le plus récent.
    
    Returns:
        Chemin du dernier modèle ou None si aucun modèle trouvé
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None
    
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        return None
    
    # Trier par date de modification
    return max(model_files, key=lambda p: p.stat().st_mtime)


# =============================================================================
# GESTION DU FEEDBACK
# =============================================================================

def save_feedback_image(
    image: Image.Image,
    true_label: str,
    predicted_label: str,
    feedback_dir: Path = FEEDBACK_DIR
) -> Path:
    """
    Sauvegarde une image de feedback avec son label.
    
    Args:
        image: Image PIL
        true_label: Label correct fourni par l'utilisateur
        predicted_label: Label prédit par le modèle
        feedback_dir: Répertoire de sauvegarde
        
    Returns:
        Chemin du fichier sauvegardé
    """
    feedback_dir = Path(feedback_dir)
    
    # Créer le sous-répertoire pour la classe
    class_dir = feedback_dir / true_label
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Générer un nom de fichier unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_pred_{predicted_label}.jpg"
    filepath = class_dir / filename
    
    # Sauvegarder l'image
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(filepath, "JPEG", quality=95)
    
    return filepath


def count_feedback_images(feedback_dir: Path = FEEDBACK_DIR) -> Dict[str, int]:
    """
    Compte le nombre d'images de feedback par classe.
    
    Returns:
        Dictionnaire {classe: nombre_images}
    """
    feedback_dir = Path(feedback_dir)
    counts = {}
    
    if not feedback_dir.exists():
        return counts
    
    for class_dir in feedback_dir.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith("."):
            counts[class_dir.name] = len(list(class_dir.glob("*.jpg")))
    
    return counts


def get_total_feedback_count(feedback_dir: Path = FEEDBACK_DIR) -> int:
    """
    Retourne le nombre total d'images de feedback.
    """
    counts = count_feedback_images(feedback_dir)
    return sum(counts.values())


# =============================================================================
# UTILITAIRES CLEARML
# =============================================================================

def setup_clearml_credentials():
    """
    Configure les credentials ClearML depuis le fichier de configuration.
    """
    # ClearML lit automatiquement le fichier clearml.conf
    # Cette fonction peut être étendue pour charger depuis des variables d'environnement
    config_path = CONFIG_DIR / "clearml.conf"
    if config_path.exists():
        os.environ.setdefault("CLEARML_CONFIG_FILE", str(config_path))


def get_device() -> str:
    """
    Retourne le device disponible (cuda ou cpu).
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# POINT D'ENTRÉE POUR TESTS
# =============================================================================

if __name__ == "__main__":
    print("=== Test des utilitaires ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Feedback dir: {FEEDBACK_DIR}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Device: {get_device()}")
    print(f"Feedback count: {get_total_feedback_count()}")
    
    # Test création modèle
    model = create_model(num_classes=5, pretrained=False)
    print(f"Modèle créé: {type(model).__name__}")
