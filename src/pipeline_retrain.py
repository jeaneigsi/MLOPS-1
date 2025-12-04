"""
pipeline_retrain.py - Pipeline ClearML pour le réentraînement automatique

Ce script définit une pipeline ClearML avec les étapes :
1. prepare_data : Téléchargement et préparation du dataset
2. train_model : Entraînement du modèle
3. register_model : Enregistrement dans le Model Registry

La pipeline peut être déclenchée par le watcher ou manuellement.

Usage:
    python src/pipeline_retrain.py
    
    # Avec un dataset spécifique:
    python src/pipeline_retrain.py --dataset-id <DATASET_ID>
    
    # Mode local (pour test):
    python src/pipeline_retrain.py --local
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from clearml import Task, Dataset, PipelineDecorator, OutputModel
from clearml.automation.controller import PipelineController


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline de réentraînement ClearML")
    parser.add_argument(
        "--dataset-id", type=str, default=None,
        help="ID du dataset ClearML à utiliser"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Nombre d'epochs d'entraînement (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Taille du batch (default: 32)"
    )
    parser.add_argument(
        "--queue", type=str, default="default",
        help="Queue ClearML pour l'exécution (default: default)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Exécuter localement (sans agent)"
    )
    return parser.parse_args()


# =============================================================================
# FONCTIONS DES ÉTAPES DE LA PIPELINE (avec décorateur)
# =============================================================================

@PipelineDecorator.component(
    cache=True,
    packages=["torch", "torchvision", "numpy", "Pillow"]
)
def prepare_data(dataset_id: str) -> dict:
    """
    Étape 1 : Prépare les données pour l'entraînement.
    
    Args:
        dataset_id: ID du dataset ClearML
        
    Returns:
        Dictionnaire avec les chemins et informations du dataset
    """
    from clearml import Dataset
    from pathlib import Path
    import json
    import os
    
    print(f"=== ÉTAPE 1: PRÉPARATION DES DONNÉES ===")
    print(f"Dataset ID: {dataset_id}")
    
    # Télécharger le dataset
    dataset = Dataset.get(dataset_id=dataset_id)
    local_path = dataset.get_local_copy()
    
    print(f"Dataset téléchargé: {local_path}")
    
    # Compter les images par classe
    data_path = Path(local_path)
    class_counts = {}
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith("."):
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            class_counts[class_dir.name] = len(images)
    
    total_images = sum(class_counts.values())
    num_classes = len(class_counts)
    
    print(f"\nDataset info:")
    print(f"  - Classes: {num_classes}")
    print(f"  - Total images: {total_images}")
    for class_name, count in sorted(class_counts.items()):
        print(f"    • {class_name}: {count}")
    
    return {
        "data_path": str(local_path),
        "num_classes": num_classes,
        "class_counts": class_counts,
        "total_images": total_images,
        "classes": list(class_counts.keys())
    }


@PipelineDecorator.component(
    cache=False,
    packages=["torch", "torchvision", "numpy", "Pillow", "tqdm"]
)
def train_model(data_info: dict, epochs: int, batch_size: int) -> dict:
    """
    Étape 2 : Entraîne le modèle.
    
    Args:
        data_info: Dictionnaire retourné par prepare_data
        epochs: Nombre d'epochs
        batch_size: Taille du batch
        
    Returns:
        Dictionnaire avec le chemin du modèle et les métriques
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms, models
    from pathlib import Path
    from tqdm import tqdm
    from datetime import datetime
    from clearml import Task
    
    print(f"\n=== ÉTAPE 2: ENTRAÎNEMENT DU MODÈLE ===")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Charger le dataset
    data_path = Path(data_info["data_path"])
    full_dataset = datasets.ImageFolder(str(data_path), transform=train_transform)
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Créer le modèle
    num_classes = data_info["num_classes"]
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    
    # Critère et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Récupérer la tâche courante pour le logging
    task = Task.current_task()
    logger = task.get_logger() if task else None
    
    # Entraînement
    best_accuracy = 0.0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(accuracy)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.2f}%")
        
        # Log dans ClearML
        if logger:
            logger.report_scalar("Training", "Loss", train_loss, epoch)
            logger.report_scalar("Validation", "Loss", val_loss, epoch)
            logger.report_scalar("Validation", "Accuracy", accuracy, epoch)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        scheduler.step()
    
    # Sauvegarder le modèle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_retrained_{timestamp}_acc{best_accuracy:.1f}.pth"
    model_path = Path("/tmp") / model_name
    
    torch.save(model.state_dict(), model_path)
    print(f"\nModèle sauvegardé: {model_path}")
    
    return {
        "model_path": str(model_path),
        "best_accuracy": best_accuracy,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "classes": data_info["classes"],
        "num_classes": num_classes
    }


@PipelineDecorator.component(
    cache=False,
    packages=["clearml"]
)
def register_model(train_result: dict, dataset_id: str) -> dict:
    """
    Étape 3 : Enregistre le modèle dans le Model Registry.
    
    Args:
        train_result: Dictionnaire retourné par train_model
        dataset_id: ID du dataset utilisé
        
    Returns:
        Dictionnaire avec les informations du modèle enregistré
    """
    from clearml import Task, OutputModel
    from datetime import datetime
    
    print(f"\n=== ÉTAPE 3: ENREGISTREMENT DU MODÈLE ===")
    
    task = Task.current_task()
    
    # Créer le modèle de sortie
    model_name = f"retrained_model_{datetime.now().strftime('%Y%m%d')}"
    
    output_model = OutputModel(
        task=task,
        name=model_name,
        framework="pytorch"
    )
    
    # Upload des poids
    output_model.update_weights(
        weights_filename=train_result["model_path"],
        auto_delete_file=False
    )
    
    # Configuration du modèle
    output_model.update_design(
        config={
            "architecture": "resnet18",
            "num_classes": train_result["num_classes"],
            "classes": train_result["classes"],
            "accuracy": train_result["best_accuracy"],
            "dataset_id": dataset_id,
            "trained_at": datetime.now().isoformat()
        }
    )
    
    # Publier le modèle
    output_model.publish()
    
    print(f"✓ Modèle enregistré dans le Registry")
    print(f"  Nom: {model_name}")
    print(f"  Accuracy: {train_result['best_accuracy']:.2f}%")
    print(f"  ID: {output_model.id}")
    
    return {
        "model_id": output_model.id,
        "model_name": model_name,
        "accuracy": train_result["best_accuracy"],
        "registered_at": datetime.now().isoformat()
    }


# =============================================================================
# PIPELINE PRINCIPALE
# =============================================================================

@PipelineDecorator.pipeline(
    name="Image_Classification_AutoRetrain",
    project="MLOps_Image_Classification",
    version="1.0"
)
def retrain_pipeline(dataset_id: str, epochs: int = 10, batch_size: int = 32):
    """
    Pipeline complète de réentraînement.
    
    Args:
        dataset_id: ID du dataset ClearML
        epochs: Nombre d'epochs
        batch_size: Taille du batch
    """
    print("=" * 60)
    print("PIPELINE DE RÉENTRAÎNEMENT")
    print("=" * 60)
    print(f"Dataset ID: {dataset_id}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Étape 1: Préparation des données
    data_info = prepare_data(dataset_id=dataset_id)
    
    # Étape 2: Entraînement
    train_result = train_model(
        data_info=data_info,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Étape 3: Enregistrement
    registry_info = register_model(
        train_result=train_result,
        dataset_id=dataset_id
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉE AVEC SUCCÈS")
    print("=" * 60)
    print(f"Modèle enregistré: {registry_info['model_name']}")
    print(f"Accuracy: {registry_info['accuracy']:.2f}%")
    print("=" * 60)
    
    return registry_info


# =============================================================================
# VERSION ALTERNATIVE AVEC PipelineController
# =============================================================================

def build_pipeline_controller(
    dataset_id: str,
    epochs: int = 10,
    batch_size: int = 32,
    queue: str = "default"
) -> PipelineController:
    """
    Construit une pipeline avec PipelineController (alternative au décorateur).
    Utile pour plus de contrôle sur l'exécution.
    """
    pipe = PipelineController(
        name="Image_Classification_AutoRetrain_v2",
        project="MLOps_Image_Classification"
    )
    
    # Paramètres globaux
    pipe.add_parameter("dataset_id", default=dataset_id)
    pipe.add_parameter("epochs", default=epochs)
    pipe.add_parameter("batch_size", default=batch_size)
    
    # Étape 1: Préparation des données
    pipe.add_function_step(
        name="prepare_data",
        function=prepare_data,
        function_kwargs={"dataset_id": "${pipeline.dataset_id}"},
        function_return=["data_info"]
    )
    
    # Étape 2: Entraînement
    pipe.add_function_step(
        name="train_model",
        function=train_model,
        function_kwargs={
            "data_info": "${prepare_data.data_info}",
            "epochs": "${pipeline.epochs}",
            "batch_size": "${pipeline.batch_size}"
        },
        function_return=["train_result"],
        parents=["prepare_data"]
    )
    
    # Étape 3: Enregistrement
    pipe.add_function_step(
        name="register_model",
        function=register_model,
        function_kwargs={
            "train_result": "${train_model.train_result}",
            "dataset_id": "${pipeline.dataset_id}"
        },
        function_return=["registry_info"],
        parents=["train_model"]
    )
    
    pipe.set_default_execution_queue(queue)
    
    return pipe


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    if not args.dataset_id:
        print("❌ Erreur: --dataset-id requis")
        print("\nUsage:")
        print("  python pipeline_retrain.py --dataset-id <ID>")
        print("\nPour obtenir l'ID du dataset:")
        print("  python dataset_versioning.py --list-datasets")
        return
    
    print("=" * 60)
    print("LANCEMENT DE LA PIPELINE DE RÉENTRAÎNEMENT")
    print("=" * 60)
    
    if args.local:
        print("Mode: LOCAL (sans agent)")
        # Exécution locale avec le décorateur
        PipelineDecorator.run_locally()
        retrain_pipeline(
            dataset_id=args.dataset_id,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        print(f"Mode: REMOTE (queue: {args.queue})")
        # Lancer sur un agent
        PipelineDecorator.set_default_execution_queue(args.queue)
        retrain_pipeline(
            dataset_id=args.dataset_id,
            epochs=args.epochs,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
