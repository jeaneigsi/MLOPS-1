"""
train_baseline.py - Entraînement initial du modèle de classification d'images

Ce script :
1. Initialise une tâche ClearML
2. Charge le dataset initial
3. Entraîne le modèle de base (ResNet pré-entraîné)
4. Log les métriques et courbes dans ClearML
5. Sauvegarde le modèle et l'enregistre dans le Model Registry

Usage:
    python src/train_baseline.py
    
    # Avec arguments personnalisés:
    python src/train_baseline.py --epochs 20 --batch-size 64
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets
from tqdm import tqdm

from clearml import Task, OutputModel, Dataset as ClearMLDataset

from utils import (
    create_model,
    save_model,
    get_train_transforms,
    get_inference_transforms,
    get_device,
    setup_clearml_credentials,
    CLEARML_PROJECT_NAME,
    CLEARML_DATASET_BASE_NAME,
    MODELS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_CLASSES
)


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement du modèle de base")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Nombre d'epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Taille du batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--model", type=str, default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Architecture du modèle (default: resnet18)"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Chemin vers le dataset local (si pas de ClearML Dataset)"
    )
    parser.add_argument(
        "--use-clearml-dataset", action="store_true",
        help="Utiliser un dataset ClearML"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Utiliser Mixed Precision Training (FP16) pour accélérer l'entraînement"
    )
    return parser.parse_args()


def load_dataset_from_clearml(dataset_name: str, project_name: str) -> Path:
    """
    Télécharge et retourne le chemin du dataset depuis ClearML.
    
    Returns:
        Chemin local vers le dataset téléchargé
    """
    print(f"Téléchargement du dataset '{dataset_name}' depuis ClearML...")
    dataset = ClearMLDataset.get(
        dataset_name=dataset_name,
        dataset_project=project_name
    )
    local_path = Path(dataset.get_local_copy())
    print(f"Dataset téléchargé: {local_path}")
    return local_path


def create_dataloaders(
    data_path: Path,
    batch_size: int,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Crée les dataloaders d'entraînement et de validation.
    
    Args:
        data_path: Chemin vers le répertoire des images (structure ImageFolder)
        batch_size: Taille du batch
        train_split: Proportion pour l'entraînement
        
    Returns:
        Tuple (train_loader, val_loader, class_names)
    """
    # Dataset complet avec transformations d'entraînement
    train_transform = get_train_transforms()
    val_transform = get_inference_transforms()
    
    # Charger le dataset pour obtenir les classes
    full_dataset = datasets.ImageFolder(str(data_path), transform=train_transform)
    class_names = full_dataset.classes
    
    # Split train/val
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Appliquer les transformations de validation au set de validation
    # Note: pour une implémentation plus propre, on devrait créer deux datasets séparés
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_names


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    task: Task,
    epoch: int,
    scaler: GradScaler = None,
    use_fp16: bool = False
) -> float:
    """
    Entraîne le modèle pendant une epoch.
    
    Args:
        scaler: GradScaler pour Mixed Precision (FP16)
        use_fp16: Si True, utilise Mixed Precision Training
    
    Returns:
        Loss moyenne de l'epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train{'⚡FP16' if use_fp16 else ''}]")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training (FP16)
        if use_fp16 and scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale loss et backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 standard
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            "loss": f"{running_loss/(batch_idx+1):.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Log dans ClearML
    task.get_logger().report_scalar("Training", "Loss", avg_loss, epoch)
    task.get_logger().report_scalar("Training", "Accuracy", accuracy, epoch)
    
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    task: Task,
    epoch: int,
    use_fp16: bool = False
) -> Tuple[float, float]:
    """
    Évalue le modèle sur le set de validation.
    
    Args:
        use_fp16: Si True, utilise autocast pour la validation
    
    Returns:
        Tuple (loss moyenne, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val{'⚡FP16' if use_fp16 else ''}]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixed Precision pour validation aussi
            if use_fp16:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Log dans ClearML
    task.get_logger().report_scalar("Validation", "Loss", avg_loss, epoch)
    task.get_logger().report_scalar("Validation", "Accuracy", accuracy, epoch)
    
    return avg_loss, accuracy


def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Configuration ClearML
    setup_clearml_credentials()
    
    # Initialiser la tâche ClearML
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="baseline_training",
        task_type=Task.TaskTypes.training
    )
    
    # Log des hyperparamètres
    task.connect({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "model_architecture": args.model,
        "precision": "FP16" if args.fp16 else "FP32"
    })
    
    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE DE BASE")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Modèle: {args.model}")
    print(f"Précision: {'⚡ FP16 (Mixed Precision)' if args.fp16 else 'FP32'}")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Charger le dataset
    if args.use_clearml_dataset:
        data_path = load_dataset_from_clearml(
            CLEARML_DATASET_BASE_NAME,
            CLEARML_PROJECT_NAME
        )
    elif args.data_path:
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {data_path}")
    else:
        # Mode démo : créer un petit dataset synthétique
        print("\n⚠️  Aucun dataset spécifié. Mode démonstration...")
        print("Pour un vrai entraînement, utilisez --data-path ou --use-clearml-dataset")
        
        # Créer un dataset de démonstration minimal
        demo_path = Path("demo_dataset")
        create_demo_dataset(demo_path, num_classes=5, images_per_class=10)
        data_path = demo_path
    
    # Créer les dataloaders
    train_loader, val_loader, class_names = create_dataloaders(
        data_path,
        args.batch_size
    )
    
    num_classes = len(class_names)
    print(f"\nClasses détectées ({num_classes}): {class_names}")
    print(f"Échantillons d'entraînement: {len(train_loader.dataset)}")
    print(f"Échantillons de validation: {len(val_loader.dataset)}")
    
    # Log des classes dans ClearML
    task.connect({"classes": class_names})
    
    # Créer le modèle
    model = create_model(
        num_classes=num_classes,
        pretrained=True,
        model_name=args.model
    )
    model.to(device)
    
    # Critère et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Mixed Precision (FP16) - GradScaler
    scaler = GradScaler() if args.fp16 and device == "cuda" else None
    use_fp16 = args.fp16 and device == "cuda"
    
    if args.fp16 and device != "cuda":
        print("⚠️  FP16 désactivé: nécessite un GPU CUDA")
    elif use_fp16:
        print("✓ Mixed Precision (FP16) activé")
    
    # Entraînement
    best_accuracy = 0.0
    best_model_path = None
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # Entraînement avec FP16
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, task, epoch,
            scaler=scaler, use_fp16=use_fp16
        )
        
        # Validation avec FP16
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device, task, epoch,
            use_fp16=use_fp16
        )
        
        # Log du learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        task.get_logger().report_scalar("Training", "Learning Rate", current_lr, epoch)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_baseline_{timestamp}_acc{val_accuracy:.1f}.pth"
            best_model_path = MODELS_DIR / model_name
            
            metadata = {
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "classes": class_names,
                "model_architecture": args.model,
                "precision": "FP16" if use_fp16 else "FP32"
            }
            
            save_model(model, best_model_path, metadata)
            print(f"✓ Nouveau meilleur modèle sauvegardé: {best_model_path}")
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    print(f"Meilleure accuracy: {best_accuracy:.2f}%")
    print(f"Modèle sauvegardé: {best_model_path}")
    
    # Enregistrer le modèle dans ClearML
    if best_model_path and best_model_path.exists():
        output_model = OutputModel(
            task=task,
            name=f"baseline_model_{args.model}",
            framework="pytorch"
        )
        output_model.update_weights(
            weights_filename=str(best_model_path),
            auto_delete_file=False
        )
        output_model.update_design(
            config_dict={
                "architecture": args.model,
                "num_classes": num_classes,
                "classes": class_names
            }
        )
        output_model.publish()
        print(f"✓ Modèle enregistré dans ClearML Model Registry")
    
    task.close()
    print("✓ Tâche ClearML terminée")


def create_demo_dataset(path: Path, num_classes: int = 2, images_per_class: int = 50):
    """
    Crée un dataset de démonstration Cats vs Dogs avec des images synthétiques.
    Utile pour tester le pipeline sans vrai dataset.
    """
    import random
    from PIL import Image, ImageDraw
    
    path = Path(path)
    class_names = ["cat", "dog"]
    
    for class_name in class_names:
        class_dir = path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for j in range(images_per_class):
            # Couleurs différentes pour chaque classe
            if class_name == "cat":
                color = (
                    random.randint(200, 255),
                    random.randint(150, 200),
                    random.randint(150, 200)
                )
            else:
                color = (
                    random.randint(150, 200),
                    random.randint(150, 200),
                    random.randint(200, 255)
                )
            
            img = Image.new("RGB", (224, 224), color)
            draw = ImageDraw.Draw(img)
            
            # Ajouter des formes pour différencier
            for _ in range(20):
                x, y = random.randint(0, 200), random.randint(0, 200)
                r = random.randint(5, 20)
                shape_color = tuple(random.randint(50, 200) for _ in range(3))
                draw.ellipse([x, y, x+r, y+r], fill=shape_color)
            
            img.save(class_dir / f"{class_name}_{j:04d}.jpg", "JPEG", quality=90)
    
    print(f"Dataset de démonstration créé: {path}")
    print(f"  - Classes: {class_names}")
    print(f"  - {images_per_class} images/classe")


if __name__ == "__main__":
    main()
