"""
download_dataset.py - Téléchargement du dataset Dogs vs Cats via KaggleHub

Dataset: princelv84/dogsvscats
Structure:
    train/cats – 12,500 images
    train/dogs – 12,500 images  
    test/cats – images for evaluation
    test/dogs – images for evaluation

Usage:
    # Télécharger et préparer le dataset
    python src/download_dataset.py
    
    # Avec upload vers ClearML
    python src/download_dataset.py --upload-clearml
    
    # Limiter le nombre d'images (pour tests)
    python src/download_dataset.py --max-images 1000
    
    # Mode démonstration (images synthétiques)
    python src/download_dataset.py --demo
"""

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm

from utils import DATA_DIR, CLEARML_PROJECT_NAME, CLEARML_DATASET_BASE_NAME


# Répertoires de destination
DATASET_DIR = DATA_DIR / "cats_vs_dogs"


def parse_args():
    parser = argparse.ArgumentParser(description="Télécharge le dataset Dogs vs Cats")
    parser.add_argument("--demo", action="store_true", help="Créer un dataset de démonstration")
    parser.add_argument("--max-images", type=int, default=None, help="Limiter le nombre d'images par classe")
    parser.add_argument("--upload-clearml", action="store_true", help="Upload vers ClearML Dataset")
    parser.add_argument("--skip-download", action="store_true", help="Ignorer le téléchargement si déjà fait")
    return parser.parse_args()


def download_with_kagglehub():
    """
    Télécharge le dataset via KaggleHub.
    
    Returns:
        Path vers le dataset téléchargé
    """
    try:
        import kagglehub
    except ImportError:
        print("❌ Package kagglehub non installé. Installation...")
        os.system("pip install kagglehub")
        import kagglehub
    
    print("📥 Téléchargement du dataset via KaggleHub...")
    print("   Dataset: princelv84/dogsvscats")
    
    # Télécharger le dataset
    path = kagglehub.dataset_download("princelv84/dogsvscats")
    
    print(f"✅ Dataset téléchargé: {path}")
    return Path(path)


def organize_dataset(source_path: Path, max_images: int = None):
    """
    Organise le dataset téléchargé dans la structure attendue.
    
    Le dataset téléchargé a la structure:
        train/cats/
        train/dogs/
        test/cats/
        test/dogs/
    
    On le copie vers data/cats_vs_dogs/ avec la même structure.
    """
    print(f"\n📂 Organisation du dataset...")
    print(f"   Source: {source_path}")
    print(f"   Destination: {DATASET_DIR}")
    
    # Vérifier la structure source
    train_cats = source_path / "train" / "cats"
    train_dogs = source_path / "train" / "dogs"
    test_cats = source_path / "test" / "cats"
    test_dogs = source_path / "test" / "dogs"
    
    # Alternative: parfois c'est "cat" au lieu de "cats"
    if not train_cats.exists():
        train_cats = source_path / "train" / "cat"
        train_dogs = source_path / "train" / "dog"
        test_cats = source_path / "test" / "cat"
        test_dogs = source_path / "test" / "dog"
    
    # Vérifier que les dossiers existent
    if not train_cats.exists():
        # Chercher la bonne structure
        print("   Recherche de la structure du dataset...")
        for p in source_path.rglob("*.jpg"):
            print(f"   Trouvé: {p.parent}")
            break
        
        # Structure alternative possible
        if (source_path / "train").exists():
            train_dir = source_path / "train"
            subdirs = list(train_dir.iterdir())
            print(f"   Sous-dossiers train: {[d.name for d in subdirs]}")
    
    # Créer la structure de destination
    dest_train_cat = DATASET_DIR / "train" / "cat"
    dest_train_dog = DATASET_DIR / "train" / "dog"
    dest_val_cat = DATASET_DIR / "val" / "cat"
    dest_val_dog = DATASET_DIR / "val" / "dog"
    
    for d in [dest_train_cat, dest_train_dog, dest_val_cat, dest_val_dog]:
        d.mkdir(parents=True, exist_ok=True)
    
    def copy_images(src_dir: Path, dest_dir: Path, max_count: int = None):
        """Copie les images d'un dossier à un autre."""
        if not src_dir.exists():
            print(f"   ⚠️ Dossier non trouvé: {src_dir}")
            return 0
        
        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))
        
        if max_count:
            images = images[:max_count]
        
        for img in tqdm(images, desc=f"   {src_dir.name} → {dest_dir.parent.name}/{dest_dir.name}"):
            shutil.copy2(img, dest_dir / img.name)
        
        return len(images)
    
    # Copier train → train
    print("\n📋 Copie des images d'entraînement...")
    train_cat_count = copy_images(train_cats, dest_train_cat, max_images)
    train_dog_count = copy_images(train_dogs, dest_train_dog, max_images)
    
    # Copier test → val
    print("\n📋 Copie des images de validation...")
    val_cat_count = copy_images(test_cats, dest_val_cat, max_images)
    val_dog_count = copy_images(test_dogs, dest_val_dog, max_images)
    
    # Résumé
    print("\n✅ Dataset organisé:")
    print(f"   📁 {DATASET_DIR}")
    print(f"   Train:")
    print(f"     - cat: {train_cat_count} images")
    print(f"     - dog: {train_dog_count} images")
    print(f"   Val:")
    print(f"     - cat: {val_cat_count} images")
    print(f"     - dog: {val_dog_count} images")
    print(f"   Total: {train_cat_count + train_dog_count + val_cat_count + val_dog_count} images")
    
    return DATASET_DIR


def create_demo_dataset(num_per_class: int = 50):
    """
    Crée un petit dataset de démonstration avec des images synthétiques.
    """
    from PIL import Image, ImageDraw
    import random
    
    print(f"\n🎨 Création du dataset de démonstration...")
    print(f"   {num_per_class} images par classe")
    
    # Créer la structure
    train_cat = DATASET_DIR / "train" / "cat"
    train_dog = DATASET_DIR / "train" / "dog"
    val_cat = DATASET_DIR / "val" / "cat"
    val_dog = DATASET_DIR / "val" / "dog"
    
    for d in [train_cat, train_dog, val_cat, val_dog]:
        d.mkdir(parents=True, exist_ok=True)
    
    train_count = int(num_per_class * 0.8)
    
    def create_image(label: str, idx: int) -> Image.Image:
        """Crée une image synthétique."""
        if label == "cat":
            bg_color = (random.randint(200, 255), random.randint(150, 200), random.randint(150, 200))
        else:
            bg_color = (random.randint(150, 200), random.randint(150, 200), random.randint(200, 255))
        
        img = Image.new('RGB', (224, 224), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Formes aléatoires
        for _ in range(30):
            x, y = random.randint(0, 200), random.randint(0, 200)
            r = random.randint(5, 25)
            color = tuple(random.randint(50, 200) for _ in range(3))
            draw.ellipse([x, y, x+r, y+r], fill=color)
        
        return img
    
    # Générer les images
    for class_name, train_dir, val_dir in [("cat", train_cat, val_cat), ("dog", train_dog, val_dog)]:
        print(f"   Génération {class_name}...")
        
        for i in range(num_per_class):
            img = create_image(class_name, i)
            
            if i < train_count:
                img.save(train_dir / f"{class_name}_{i:04d}.jpg", "JPEG", quality=90)
            else:
                img.save(val_dir / f"{class_name}_{i:04d}.jpg", "JPEG", quality=90)
    
    print(f"\n✅ Dataset de démonstration créé:")
    print(f"   📁 {DATASET_DIR}")
    print(f"   Train: {train_count * 2} images")
    print(f"   Val: {(num_per_class - train_count) * 2} images")
    
    return DATASET_DIR


def upload_to_clearml(dataset_path: Path):
    """Upload le dataset vers ClearML."""
    from clearml import Dataset, Task
    
    print("\n☁️ Upload vers ClearML...")
    
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="dataset_upload",
        task_type=Task.TaskTypes.data_processing
    )
    
    dataset = Dataset.create(
        dataset_name=CLEARML_DATASET_BASE_NAME,
        dataset_project=CLEARML_PROJECT_NAME,
        description="Cats vs Dogs - Dataset de base (princelv84/dogsvscats)"
    )
    
    dataset.add_files(path=str(dataset_path))
    
    # Compter les images
    train_count = len(list((dataset_path / "train").rglob("*.jpg")))
    val_count = len(list((dataset_path / "val").rglob("*.jpg")))
    
    dataset.set_metadata({
        "type": "base",
        "classes": ["cat", "dog"],
        "train_images": train_count,
        "val_images": val_count,
        "source": "kaggle:princelv84/dogsvscats"
    })
    
    print("   Upload en cours...")
    dataset.upload()
    dataset.finalize()
    
    print(f"\n✅ Dataset uploadé vers ClearML:")
    print(f"   ID: {dataset.id}")
    print(f"   Nom: {CLEARML_DATASET_BASE_NAME}")
    
    task.close()
    return dataset.id


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🐱🐕 PRÉPARATION DU DATASET CATS VS DOGS")
    print("=" * 60)
    
    dataset_path = None
    
    if args.demo:
        dataset_path = create_demo_dataset(num_per_class=50)
    else:
        # Vérifier si déjà téléchargé
        if args.skip_download and DATASET_DIR.exists():
            print(f"✓ Dataset existant: {DATASET_DIR}")
            dataset_path = DATASET_DIR
        else:
            # Télécharger via KaggleHub
            source_path = download_with_kagglehub()
            dataset_path = organize_dataset(source_path, max_images=args.max_images)
    
    if dataset_path and args.upload_clearml:
        upload_to_clearml(dataset_path)
    
    print("\n" + "=" * 60)
    print("✅ TERMINÉ")
    print("=" * 60)
    print(f"\nPour entraîner le modèle:")
    print(f"  python src/train_baseline.py --data-path {dataset_path}")


if __name__ == "__main__":
    main()
