# MLOps Cats vs Dogs Classification with ClearML 🐱🐕

Projet MLOps complet pour la classification **Cats vs Dogs** avec :
- 🔄 Boucle de feedback utilisateur via Gradio
- 📦 Versioning automatique des datasets
- 🚀 Pipeline de réentraînement automatique
- 📊 Suivi des expériences avec ClearML

## 📁 Structure du projet

```
mlops_clearml_project/
├── config/
│   └── clearml.conf              # Configuration ClearML
├── data/
│   ├── cats_vs_dogs/             # Dataset principal
│   │   ├── train/
│   │   │   ├── cat/
│   │   │   └── dog/
│   │   └── val/
│   │       ├── cat/
│   │       └── dog/
│   └── feedback_labeled/         # Images annotées par les utilisateurs
├── models/                       # Modèles sauvegardés localement
├── src/
│   ├── utils.py                  # Fonctions utilitaires partagées
│   ├── download_dataset.py       # Téléchargement dataset Cats vs Dogs
│   ├── train_baseline.py         # Entraînement initial du modèle
│   ├── gradio_app.py             # Interface Gradio + feedback
│   ├── dataset_versioning.py     # Gestion des versions de dataset
│   ├── pipeline_retrain.py       # Pipeline ClearML de réentraînement
│   └── watcher_trigger.py        # Détection et déclenchement auto
└── requirements.txt
```

## 🚀 Installation

```bash
# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## ⚙️ Configuration ClearML

La configuration est déjà présente dans `config/clearml.conf`. 
Alternativement, vous pouvez exécuter :

```bash
clearml-init
```

## 📋 Workflow MLOps

### 1️⃣ Entraînement initial

```bash
# Mode démo (sans dataset)
python src/train_baseline.py

# Avec un dataset local
python src/train_baseline.py --data-path data/mon_dataset

# Avec un dataset ClearML
python src/train_baseline.py --use-clearml-dataset
```

### 2️⃣ Lancer l'interface Gradio

```bash
# Utilise le dernier modèle local
python src/gradio_app.py

# Avec un modèle spécifique
python src/gradio_app.py --model-path models/best_model.pth

# Depuis le Model Registry ClearML
python src/gradio_app.py --use-registry

# Avec partage public
python src/gradio_app.py --share
```

L'interface permet de :
- 📤 Uploader une image
- 🔍 Obtenir une prédiction
- ✅ Corriger la prédiction (feedback)

### 3️⃣ Créer une nouvelle version du dataset

```bash
# Créer le dataset de base initial
python src/dataset_versioning.py --create-base --data-path data/initial

# Visualiser les feedbacks collectés
python src/dataset_versioning.py

# Créer une nouvelle version avec les feedbacks
python src/dataset_versioning.py --add-feedback

# Lister tous les datasets
python src/dataset_versioning.py --list-datasets
```

### 4️⃣ Lancer la pipeline de réentraînement

```bash
# Avec un dataset spécifique (mode local)
python src/pipeline_retrain.py --dataset-id <DATASET_ID> --local

# Sur un agent ClearML
python src/pipeline_retrain.py --dataset-id <DATASET_ID> --queue default
```

### 5️⃣ Surveillance automatique

```bash
# Vérification unique
python src/watcher_trigger.py

# Mode surveillance continue (toutes les 5 min)
python src/watcher_trigger.py --watch --interval 300

# Forcer le déclenchement
python src/watcher_trigger.py --force
```

## 🔁 Boucle MLOps complète

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   1. train_baseline.py                                      │
│      └─→ Entraîne le modèle initial                        │
│          └─→ Enregistre dans Model Registry                │
│                                                             │
│   2. gradio_app.py                                          │
│      └─→ Interface de prédiction                           │
│          └─→ Collecte les feedbacks (corrections)          │
│              └─→ Sauvegarde dans data/feedback_labeled/    │
│                                                             │
│   3. dataset_versioning.py                                  │
│      └─→ Crée nouvelle version du dataset                  │
│          └─→ Upload vers ClearML                           │
│                                                             │
│   4. watcher_trigger.py                                     │
│      └─→ Détecte le nouveau dataset                        │
│          └─→ Déclenche la pipeline                         │
│                                                             │
│   5. pipeline_retrain.py                                    │
│      └─→ Prépare les données                               │
│          └─→ Entraîne le modèle                            │
│              └─→ Enregistre dans Model Registry            │
│                                                             │
│   ↺ Retour à l'étape 2 avec le nouveau modèle              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Accès ClearML

- **Web UI**: https://app.clear.ml/
- **Projet**: MLOps_Image_Classification

Dans l'interface ClearML, vous pouvez visualiser :
- Les expériences et métriques d'entraînement
- Les versions de datasets
- Les modèles dans le Model Registry
- Les pipelines et leur statut

## 🛠️ Personnalisation

### Modifier les classes

Dans `src/utils.py`, modifiez `DEFAULT_CLASSES` :

```python
DEFAULT_CLASSES = ["chat", "chien", "oiseau", "poisson", "lapin"]
```

### Changer l'architecture du modèle

```bash
python src/train_baseline.py --model resnet50
```

### Ajuster les hyperparamètres

```bash
python src/train_baseline.py --epochs 20 --batch-size 64 --lr 0.0001
```

## 📝 Pour le rapport

Ce projet démontre une chaîne MLOps complète avec :

1. **Entraînement initial** avec logging ClearML
2. **Boucle de feedback** via interface Gradio
3. **Versioning des données** avec ClearML Dataset
4. **Déclenchement automatique** via watcher
5. **Pipeline orchestrée** avec ClearML Pipeline
6. **Model Registry** pour la gestion des versions

> "Dans une optique plus proche d'un projet industriel, nous avons choisi de structurer la chaîne MLOps en scripts Python plutôt qu'en notebooks. Chaque script correspond à un bloc fonctionnel, ce qui facilite la réutilisation, l'orchestration et le déploiement."
