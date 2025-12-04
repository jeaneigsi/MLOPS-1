"""
watcher_trigger.py - Surveillance et déclenchement automatique de la pipeline

Ce script :
1. Surveille les nouvelles versions de dataset dans ClearML
2. Détecte les changements par rapport à la dernière version traitée
3. Déclenche automatiquement la pipeline de réentraînement

Peut être exécuté :
- Manuellement (python watcher_trigger.py)
- Via un cron job
- Comme tâche ClearML périodique

Usage:
    # Vérification unique:
    python src/watcher_trigger.py
    
    # Mode surveillance continue:
    python src/watcher_trigger.py --watch --interval 300
    
    # Force le déclenchement:
    python src/watcher_trigger.py --force
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from clearml import Task, Dataset

from utils import (
    setup_clearml_credentials,
    CLEARML_PROJECT_NAME,
    CLEARML_DATASET_FEEDBACK_NAME,
    PROJECT_ROOT
)


# Fichier pour stocker l'état du dernier dataset traité
STATE_FILE = PROJECT_ROOT / ".watcher_state.json"


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Watcher pour déclenchement automatique de pipeline")
    
    parser.add_argument(
        "--watch", action="store_true",
        help="Mode surveillance continue"
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Intervalle de vérification en secondes (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forcer le déclenchement même si pas de nouveau dataset"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Afficher ce qui serait fait sans exécuter"
    )
    parser.add_argument(
        "--queue", type=str, default="default",
        help="Queue ClearML pour la pipeline (default: default)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Nombre d'epochs pour le réentraînement (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Taille du batch (default: 32)"
    )
    
    return parser.parse_args()


# =============================================================================
# GESTION DE L'ÉTAT
# =============================================================================

def load_state() -> Dict[str, Any]:
    """
    Charge l'état du watcher depuis le fichier JSON.
    
    Returns:
        Dictionnaire avec l'état
    """
    if not STATE_FILE.exists():
        return {
            "last_processed_id": None,
            "last_processed_name": None,
            "last_check": None,
            "history": []
        }
    
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]):
    """
    Sauvegarde l'état du watcher.
    
    Args:
        state: Dictionnaire d'état à sauvegarder
    """
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def update_state(dataset_id: str, dataset_name: str, triggered: bool = True):
    """
    Met à jour l'état après traitement d'un dataset.
    
    Args:
        dataset_id: ID du dataset traité
        dataset_name: Nom du dataset
        triggered: Si True, une pipeline a été déclenchée
    """
    state = load_state()
    
    state["last_processed_id"] = dataset_id
    state["last_processed_name"] = dataset_name
    state["last_check"] = datetime.now().isoformat()
    
    # Ajouter à l'historique (garder les 10 derniers)
    state["history"].insert(0, {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "triggered": triggered
    })
    state["history"] = state["history"][:10]
    
    save_state(state)


# =============================================================================
# DÉTECTION DES DATASETS
# =============================================================================

def get_all_feedback_datasets() -> List[Dict[str, Any]]:
    """
    Récupère tous les datasets de feedback du projet.
    
    Returns:
        Liste de datasets triés par date (plus récent en premier)
    """
    try:
        datasets = Dataset.list_datasets(
            dataset_project=CLEARML_PROJECT_NAME,
            partial_name=CLEARML_DATASET_FEEDBACK_NAME,
            only_completed=True
        )
        
        # Trier par date de création
        sorted_datasets = sorted(
            datasets,
            key=lambda d: d.get("created", ""),
            reverse=True
        )
        
        return sorted_datasets
        
    except Exception as e:
        print(f"Erreur lors de la récupération des datasets: {e}")
        return []


def get_latest_dataset() -> Optional[Dict[str, Any]]:
    """
    Récupère le dataset le plus récent.
    
    Returns:
        Dataset ou None
    """
    datasets = get_all_feedback_datasets()
    return datasets[0] if datasets else None


def check_for_new_dataset() -> Optional[Dict[str, Any]]:
    """
    Vérifie s'il y a un nouveau dataset depuis le dernier traitement.
    
    Returns:
        Nouveau dataset ou None
    """
    state = load_state()
    last_processed_id = state.get("last_processed_id")
    
    latest = get_latest_dataset()
    
    if not latest:
        return None
    
    if latest["id"] != last_processed_id:
        return latest
    
    return None


# =============================================================================
# DÉCLENCHEMENT DE LA PIPELINE
# =============================================================================

def trigger_pipeline(
    dataset_id: str,
    queue: str = "default",
    epochs: int = 10,
    batch_size: int = 32,
    dry_run: bool = False
) -> Optional[str]:
    """
    Déclenche la pipeline de réentraînement.
    
    Args:
        dataset_id: ID du dataset
        queue: Queue ClearML
        epochs: Nombre d'epochs
        batch_size: Taille du batch
        dry_run: Si True, n'exécute pas vraiment
        
    Returns:
        ID de la tâche pipeline ou None
    """
    print(f"\n🚀 Déclenchement de la pipeline de réentraînement")
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Queue: {queue}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    if dry_run:
        print("\n   [DRY RUN] Aucune action effectuée")
        return None
    
    try:
        # Option 1: Importer et appeler directement le module
        from pipeline_retrain import retrain_pipeline
        from clearml.automation import PipelineDecorator
        
        # Configurer la queue
        PipelineDecorator.set_default_execution_queue(queue)
        
        # Lancer la pipeline
        result = retrain_pipeline(
            dataset_id=dataset_id,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print(f"\n✓ Pipeline déclenchée avec succès")
        return result.get("model_id") if result else None
        
    except Exception as e:
        print(f"\n❌ Erreur lors du déclenchement: {e}")
        
        # Option 2: Alternative via Task.enqueue
        try:
            print("Tentative alternative via Task.enqueue...")
            
            # Cloner et exécuter une tâche existante
            # (nécessite qu'une pipeline ait déjà été exécutée une fois)
            existing_tasks = Task.get_tasks(
                project_name=CLEARML_PROJECT_NAME,
                task_name="Image_Classification_AutoRetrain"
            )
            
            if existing_tasks:
                cloned = Task.clone(task=existing_tasks[0].id)
                cloned.set_parameters({"dataset_id": dataset_id})
                Task.enqueue(task=cloned, queue_name=queue)
                print(f"✓ Tâche clonée et mise en queue: {cloned.id}")
                return cloned.id
            else:
                print("❌ Aucune tâche pipeline existante trouvée")
                return None
                
        except Exception as e2:
            print(f"❌ Alternative échouée: {e2}")
            return None


# =============================================================================
# BOUCLE DE SURVEILLANCE
# =============================================================================

def run_check(args) -> bool:
    """
    Effectue une vérification unique.
    
    Returns:
        True si une pipeline a été déclenchée
    """
    print(f"\n{'='*60}")
    print(f"VÉRIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Charger l'état actuel
    state = load_state()
    last_id = state.get("last_processed_id", "Aucun")
    print(f"Dernier dataset traité: {last_id}")
    
    if args.force:
        print("\n⚠️  Mode forcé: récupération du dernier dataset...")
        new_dataset = get_latest_dataset()
    else:
        new_dataset = check_for_new_dataset()
    
    if not new_dataset:
        print("\n✓ Pas de nouveau dataset détecté")
        return False
    
    print(f"\n🆕 Nouveau dataset détecté!")
    print(f"   ID: {new_dataset['id']}")
    print(f"   Nom: {new_dataset['name']}")
    print(f"   Créé: {new_dataset.get('created', 'N/A')}")
    
    # Déclencher la pipeline
    task_id = trigger_pipeline(
        dataset_id=new_dataset["id"],
        queue=args.queue,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    # Mettre à jour l'état
    if not args.dry_run:
        update_state(
            dataset_id=new_dataset["id"],
            dataset_name=new_dataset["name"],
            triggered=task_id is not None
        )
    
    return True


def run_watch_loop(args):
    """
    Exécute la boucle de surveillance continue.
    """
    print(f"\n🔄 Mode surveillance activé")
    print(f"   Intervalle: {args.interval} secondes")
    print(f"   Appuyez sur Ctrl+C pour arrêter\n")
    
    try:
        while True:
            triggered = run_check(args)
            
            if triggered:
                print(f"\n⏳ Attente de {args.interval} secondes avant prochaine vérification...")
            else:
                print(f"⏳ Prochaine vérification dans {args.interval} secondes...")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt de la surveillance")


# =============================================================================
# AFFICHAGE DU STATUT
# =============================================================================

def show_status():
    """Affiche le statut actuel du watcher."""
    state = load_state()
    
    print("\n📊 STATUT DU WATCHER")
    print("=" * 40)
    print(f"Dernier dataset traité: {state.get('last_processed_name', 'Aucun')}")
    print(f"ID: {state.get('last_processed_id', 'N/A')}")
    print(f"Dernière vérification: {state.get('last_check', 'Jamais')}")
    
    history = state.get("history", [])
    if history:
        print(f"\nHistorique ({len(history)} derniers):")
        for entry in history[:5]:
            triggered = "✓" if entry.get("triggered") else "○"
            print(f"  {triggered} {entry['dataset_name']} - {entry['timestamp'][:19]}")
    
    # Datasets disponibles
    print("\n📁 Datasets disponibles:")
    datasets = get_all_feedback_datasets()
    if datasets:
        for ds in datasets[:5]:
            marker = "→" if ds["id"] == state.get("last_processed_id") else " "
            print(f"  {marker} {ds['name']} ({ds['id'][:8]}...)")
    else:
        print("  Aucun dataset trouvé")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Configuration ClearML
    setup_clearml_credentials()
    
    # Initialiser une tâche ClearML si en mode surveillance
    if args.watch:
        task = Task.init(
            project_name=CLEARML_PROJECT_NAME,
            task_name="dataset_watcher",
            task_type=Task.TaskTypes.monitor
        )
    else:
        task = None
    
    print("=" * 60)
    print("WATCHER - DÉTECTION DE NOUVEAUX DATASETS")
    print("=" * 60)
    
    # Afficher le statut
    show_status()
    
    if args.watch:
        run_watch_loop(args)
    else:
        run_check(args)
    
    if task:
        task.close()


if __name__ == "__main__":
    main()
