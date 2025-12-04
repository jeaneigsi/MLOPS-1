"""
dataset_versioning.py - Gestion des versions de datasets ClearML

Ce script :
1. Récupère les images de feedback collectées
2. Crée une nouvelle version du dataset avec ClearML
3. Maintient la traçabilité via les datasets parents

Usage:
    python src/dataset_versioning.py
    
    # Créer le dataset initial:
    python src/dataset_versioning.py --create-base --data-path data/initial_dataset
    
    # Créer une nouvelle version avec le feedback:
    python src/dataset_versioning.py --add-feedback
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from clearml import Task, Dataset

from utils import (
    setup_clearml_credentials,
    count_feedback_images,
    get_total_feedback_count,
    CLEARML_PROJECT_NAME,
    CLEARML_DATASET_BASE_NAME,
    CLEARML_DATASET_FEEDBACK_NAME,
    FEEDBACK_DIR,
    DATA_DIR
)


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Gestion des versions de datasets ClearML")
    
    parser.add_argument(
        "--create-base", action="store_true",
        help="Créer le dataset de base initial"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Chemin vers les données pour le dataset de base"
    )
    parser.add_argument(
        "--add-feedback", action="store_true",
        help="Créer une nouvelle version avec les feedbacks collectés"
    )
    parser.add_argument(
        "--min-feedback", type=int, default=10,
        help="Nombre minimum de feedbacks requis pour créer une nouvelle version (default: 10)"
    )
    parser.add_argument(
        "--clear-feedback", action="store_true",
        help="Effacer les feedbacks locaux après upload"
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="Lister tous les datasets du projet"
    )
    
    return parser.parse_args()


def create_base_dataset(data_path: Path) -> Dataset:
    """
    Crée le dataset de base initial.
    
    Args:
        data_path: Chemin vers le répertoire des images initiales
        
    Returns:
        Dataset ClearML créé
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Répertoire non trouvé: {data_path}")
    
    print(f"Création du dataset de base depuis: {data_path}")
    
    # Compter les fichiers
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.rglob(f"*{ext}"))
        image_files.extend(data_path.rglob(f"*{ext.upper()}"))
    
    print(f"Images trouvées: {len(image_files)}")
    
    # Créer le dataset ClearML
    dataset = Dataset.create(
        dataset_name=CLEARML_DATASET_BASE_NAME,
        dataset_project=CLEARML_PROJECT_NAME,
        description="Dataset initial pour la classification d'images"
    )
    
    # Ajouter les fichiers
    dataset.add_files(path=str(data_path))
    
    # Métadonnées
    dataset.set_metadata({
        "type": "base",
        "created_at": datetime.now().isoformat(),
        "image_count": len(image_files),
        "source_path": str(data_path)
    })
    
    # Upload et finalisation
    dataset.upload()
    dataset.finalize()
    
    print(f"✓ Dataset de base créé avec succès!")
    print(f"  ID: {dataset.id}")
    print(f"  Nom: {CLEARML_DATASET_BASE_NAME}")
    print(f"  Images: {len(image_files)}")
    
    return dataset


def get_latest_dataset(dataset_name: str) -> Optional[Dataset]:
    """
    Récupère la dernière version d'un dataset.
    
    Returns:
        Dataset ou None si non trouvé
    """
    try:
        datasets = Dataset.list_datasets(
            dataset_project=CLEARML_PROJECT_NAME,
            partial_name=dataset_name,
            only_completed=True
        )
        
        if not datasets:
            return None
        
        # Trier par date de création (la plus récente en premier)
        sorted_datasets = sorted(
            datasets,
            key=lambda d: d.get("created", ""),
            reverse=True
        )
        
        # Récupérer le dataset complet
        latest = Dataset.get(dataset_id=sorted_datasets[0]["id"])
        return latest
        
    except Exception as e:
        print(f"Erreur lors de la recherche du dataset: {e}")
        return None


def create_feedback_version(
    min_feedback: int = 10,
    clear_after_upload: bool = False
) -> Optional[Dataset]:
    """
    Crée une nouvelle version du dataset avec les feedbacks collectés.
    
    Args:
        min_feedback: Nombre minimum de feedbacks requis
        clear_after_upload: Si True, supprime les feedbacks locaux après upload
        
    Returns:
        Nouveau dataset ou None si pas assez de feedback
    """
    # Vérifier le nombre de feedbacks
    feedback_count = get_total_feedback_count()
    print(f"Feedbacks collectés: {feedback_count}")
    
    if feedback_count < min_feedback:
        print(f"❌ Pas assez de feedbacks ({feedback_count} < {min_feedback})")
        print(f"   Collectez au moins {min_feedback - feedback_count} feedbacks supplémentaires.")
        return None
    
    # Statistiques par classe
    stats = count_feedback_images()
    print("\nDistribution par classe:")
    for class_name, count in sorted(stats.items()):
        print(f"  • {class_name}: {count}")
    
    # Récupérer le dernier dataset (base ou feedback précédent)
    parent_dataset = get_latest_dataset(CLEARML_DATASET_FEEDBACK_NAME)
    if not parent_dataset:
        parent_dataset = get_latest_dataset(CLEARML_DATASET_BASE_NAME)
    
    if not parent_dataset:
        print("⚠️  Aucun dataset parent trouvé. Création sans parent.")
        parent_ids = []
    else:
        print(f"\nDataset parent: {parent_dataset.name} (ID: {parent_dataset.id})")
        parent_ids = [parent_dataset.id]
    
    # Générer un nom de version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"{CLEARML_DATASET_FEEDBACK_NAME}_v{timestamp}"
    
    print(f"\nCréation du nouveau dataset: {version_name}")
    
    # Créer le nouveau dataset
    dataset = Dataset.create(
        dataset_name=version_name,
        dataset_project=CLEARML_PROJECT_NAME,
        parent_datasets=parent_ids,
        description=f"Dataset avec {feedback_count} feedbacks utilisateur"
    )
    
    # Ajouter les fichiers de feedback
    dataset.add_files(path=str(FEEDBACK_DIR))
    
    # Métadonnées
    dataset.set_metadata({
        "type": "feedback",
        "created_at": datetime.now().isoformat(),
        "feedback_count": feedback_count,
        "class_distribution": stats,
        "parent_id": parent_ids[0] if parent_ids else None
    })
    
    # Upload et finalisation
    print("Upload en cours...")
    dataset.upload()
    dataset.finalize()
    
    print(f"\n✓ Dataset créé avec succès!")
    print(f"  ID: {dataset.id}")
    print(f"  Nom: {version_name}")
    print(f"  Feedbacks ajoutés: {feedback_count}")
    
    # Optionnellement nettoyer les feedbacks locaux
    if clear_after_upload:
        clear_local_feedback()
    
    return dataset


def clear_local_feedback():
    """Supprime les feedbacks locaux après upload."""
    feedback_path = Path(FEEDBACK_DIR)
    if feedback_path.exists():
        for class_dir in feedback_path.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith("."):
                shutil.rmtree(class_dir)
        print("✓ Feedbacks locaux supprimés")


def list_all_datasets():
    """Liste tous les datasets du projet."""
    print(f"\nDatasets du projet '{CLEARML_PROJECT_NAME}':")
    print("=" * 60)
    
    datasets = Dataset.list_datasets(
        dataset_project=CLEARML_PROJECT_NAME,
        only_completed=True
    )
    
    if not datasets:
        print("Aucun dataset trouvé.")
        return
    
    # Trier par date
    sorted_datasets = sorted(
        datasets,
        key=lambda d: d.get("created", ""),
        reverse=True
    )
    
    for i, ds in enumerate(sorted_datasets, 1):
        created = ds.get("created", "N/A")[:19]  # Tronquer la date
        name = ds.get("name", "N/A")
        ds_id = ds.get("id", "N/A")[:8]  # Tronquer l'ID
        
        print(f"{i}. {name}")
        print(f"   ID: {ds_id}... | Créé: {created}")


def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Configuration ClearML
    setup_clearml_credentials()
    
    # Initialiser une tâche ClearML
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="dataset_versioning",
        task_type=Task.TaskTypes.data_processing
    )
    
    print("=" * 60)
    print("GESTION DES VERSIONS DE DATASETS")
    print("=" * 60)
    
    if args.list_datasets:
        list_all_datasets()
        
    elif args.create_base:
        if not args.data_path:
            print("❌ Erreur: --data-path requis avec --create-base")
            return
        create_base_dataset(Path(args.data_path))
        
    elif args.add_feedback:
        create_feedback_version(
            min_feedback=args.min_feedback,
            clear_after_upload=args.clear_feedback
        )
    
    else:
        # Comportement par défaut : afficher les stats
        print(f"\nFeedbacks locaux: {get_total_feedback_count()}")
        stats = count_feedback_images()
        if stats:
            print("\nDistribution:")
            for class_name, count in sorted(stats.items()):
                print(f"  • {class_name}: {count}")
        
        print("\nCommandes disponibles:")
        print("  --create-base --data-path <path>  : Créer le dataset initial")
        print("  --add-feedback                     : Créer une version avec les feedbacks")
        print("  --list-datasets                    : Lister tous les datasets")
    
    task.close()


if __name__ == "__main__":
    main()
