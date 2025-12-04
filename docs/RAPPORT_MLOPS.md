# Rapport de Projet MLOps
## Classification d'Images Cats vs Dogs avec ClearML

---

**Auteur :** Jean Olivier Lompo  
**Date :** 4 Décembre 2024  
**Cours :** MLOps - Machine Learning Operations

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Architecture du Système](#2-architecture-du-système)
3. [Composants MLOps](#3-composants-mlops)
4. [Implémentation Technique](#4-implémentation-technique)
5. [Pipeline de Réentraînement](#5-pipeline-de-réentraînement)
6. [Résultats et Métriques](#6-résultats-et-métriques)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction

### 1.1 Contexte

Ce projet implémente une chaîne MLOps complète pour la classification binaire d'images (chats vs chiens) en utilisant **ClearML** comme plateforme d'orchestration. L'objectif est de démontrer les bonnes pratiques en matière de Machine Learning Operations.

### 1.2 Objectifs

Les objectifs principaux de ce projet sont :

- Mettre en place un **pipeline d'entraînement** automatisé
- Implémenter une **boucle de feedback** utilisateur via interface web
- Gérer le **versioning des datasets** de manière traçable
- Automatiser le **réentraînement** lors de nouvelles données
- Utiliser un **Model Registry** pour la gestion des versions de modèles

### 1.3 Technologies Utilisées

| Technologie | Rôle |
|-------------|------|
| **Python 3.11** | Langage de programmation |
| **PyTorch** | Framework de Deep Learning |
| **ClearML** | Plateforme MLOps (tracking, pipelines, registry) |
| **Gradio** | Interface utilisateur web |
| **ResNet-18** | Architecture du modèle CNN |

---

## 2. Architecture du Système

### 2.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE MLOPS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐      │
│   │   Dataset    │────▶│  Training    │────▶│    Model     │      │
│   │   ClearML    │     │   Script     │     │   Registry   │      │
│   └──────────────┘     └──────────────┘     └──────────────┘      │
│          ▲                                         │               │
│          │                                         ▼               │
│   ┌──────────────┐                         ┌──────────────┐       │
│   │   Dataset    │                         │   Gradio     │       │
│   │  Versioning  │◀────────────────────────│   App        │       │
│   └──────────────┘      Feedback           └──────────────┘       │
│          ▲                                         │               │
│          │                                         │               │
│   ┌──────────────┐     ┌──────────────┐          │               │
│   │   Watcher    │────▶│   Pipeline   │◀─────────┘               │
│   │   Trigger    │     │   Retrain    │                          │
│   └──────────────┘     └──────────────┘                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Structure du Projet

```text
mlops_clearml_project/
├── config/
│   └── clearml.conf              # Configuration API ClearML
├── data/
│   ├── cats_vs_dogs/             # Dataset principal
│   │   ├── train/                # 25,000 images
│   │   │   ├── cat/              # 12,500 images
│   │   │   └── dog/              # 12,500 images
│   │   └── val/                  # Images de validation
│   └── feedback_labeled/         # Corrections utilisateurs
├── models/                       # Checkpoints locaux
├── src/
│   ├── utils.py                  # Fonctions utilitaires
│   ├── download_dataset.py       # Téléchargement dataset
│   ├── train_baseline.py         # Entraînement initial
│   ├── gradio_app.py             # Interface web
│   ├── dataset_versioning.py     # Gestion versions
│   ├── pipeline_retrain.py       # Pipeline ClearML
│   └── watcher_trigger.py        # Détection automatique
└── requirements.txt
```

---

## 3. Composants MLOps

### 3.1 Entraînement Initial (`train_baseline.py`)

Le script d'entraînement de base utilise un modèle **ResNet-18** pré-entraîné sur ImageNet, avec fine-tuning pour la classification binaire.

**Formule de la fonction de perte (Cross-Entropy) :**

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Où :
- $N$ = nombre d'échantillons
- $C$ = nombre de classes (2 pour cats/dogs)
- $y_{i,c}$ = label réel (one-hot)
- $\hat{y}_{i,c}$ = probabilité prédite

**Hyperparamètres par défaut :**

| Paramètre | Valeur |
|-----------|--------|
| Learning Rate | $\alpha = 0.001$ |
| Batch Size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Scheduler | StepLR (step=5, $\gamma=0.5$) |

**Mise à jour des poids (Adam) :**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

### 3.2 Interface Utilisateur (`gradio_app.py`)

L'interface Gradio permet :

1. **Prédiction** : Upload d'image → Classification
2. **Feedback** : Correction du label si erreur
3. **Statistiques** : Visualisation des feedbacks collectés

**Flux de prédiction :**

$$\text{Image} \xrightarrow{\text{Preprocess}} \text{Tensor}_{224 \times 224} \xrightarrow{\text{ResNet}} \text{Logits} \xrightarrow{\text{Softmax}} P(\text{cat}), P(\text{dog})$$

**Normalisation des images :**

$$x_{norm} = \frac{x - \mu}{\sigma}$$

Avec $\mu = [0.485, 0.456, 0.406]$ et $\sigma = [0.229, 0.224, 0.225]$ (statistiques ImageNet).

#### 3.2.1 Extension Dynamique des Classes (Nouveau Label)

Une fonctionnalité clé de l'interface est la possibilité d'**ajouter de nouvelles classes** au modèle via le champ "Nouveau label". Cela permet le **Continuous Learning** :

**Cas d'usage :**
- Le modèle initial est entraîné sur 2 classes : `cat` et `dog`
- Un utilisateur uploade une image de **lapin** 🐰
- Le modèle prédit "cat" (classe la plus proche)
- L'utilisateur corrige en saisissant "rabbit" dans le champ "Nouveau label"
- L'image est sauvegardée dans `data/feedback_labeled/rabbit/`

**Structure résultante :**

```
data/feedback_labeled/
├── cat/           # Corrections vers "cat"
├── dog/           # Corrections vers "dog"
└── rabbit/        # 🆕 Nouvelle classe créée dynamiquement
```

**Évolution du modèle :**

$$\text{Classes}_{v1} = \{\text{cat}, \text{dog}\} \xrightarrow{\text{feedback}} \text{Classes}_{v2} = \{\text{cat}, \text{dog}, \text{rabbit}\}$$

Lors du réentraînement, le modèle adapte automatiquement sa couche de sortie :

$$\text{FC}: 512 \rightarrow 2 \quad \Rightarrow \quad \text{FC}: 512 \rightarrow 3$$

Cette approche permet une **amélioration continue** du modèle sans intervention manuelle sur le code.

### 3.3 Versioning des Données (`dataset_versioning.py`)

Le versioning ClearML permet de :

- Créer des **versions incrémentales** du dataset
- Maintenir la **traçabilité** via les datasets parents
- **Automatiser** l'upload des feedbacks collectés

**Schéma de versioning :**

```
cats_vs_dogs_base (v1)
        │
        ▼
cats_vs_dogs_feedback_v20241204_120000 (v2)
        │
        ▼
cats_vs_dogs_feedback_v20241205_150000 (v3)
        │
        ▼
       ...
```

### 3.4 Pipeline de Réentraînement (`pipeline_retrain.py`)

La pipeline ClearML se compose de 3 étapes :

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  prepare_data  │────▶│  train_model   │────▶│ register_model │
└────────────────┘     └────────────────┘     └────────────────┘
```

**Étape 1 - Préparation des données :**
- Téléchargement du dataset depuis ClearML
- Analyse de la distribution des classes

**Étape 2 - Entraînement :**
- Fine-tuning du modèle
- Logging des métriques (loss, accuracy)

**Étape 3 - Enregistrement :**
- Upload du modèle dans le Model Registry
- Métadonnées (accuracy, classes, architecture)

### 3.5 Watcher Automatique (`watcher_trigger.py`)

Le watcher surveille les nouveaux datasets et déclenche automatiquement la pipeline :

```python
if latest_dataset_id != last_processed_id:
    trigger_pipeline(latest_dataset_id)
    save_state(latest_dataset_id)
```

---

## 4. Implémentation Technique

### 4.1 Modèle de Classification

**Architecture ResNet-18 modifiée :**

| Couche | Entrée | Sortie |
|--------|--------|--------|
| Conv1 + BN + ReLU + MaxPool | 3×224×224 | 64×56×56 |
| Layer1 (2 BasicBlocks) | 64×56×56 | 64×56×56 |
| Layer2 (2 BasicBlocks) | 64×56×56 | 128×28×28 |
| Layer3 (2 BasicBlocks) | 128×28×28 | 256×14×14 |
| Layer4 (2 BasicBlocks) | 256×14×14 | 512×7×7 |
| AdaptiveAvgPool | 512×7×7 | 512×1×1 |
| **FC (modifiée)** | 512 | **2** |

**Nombre de paramètres :**

$$\text{Total} \approx 11.2M \text{ paramètres}$$

### 4.2 Augmentation de Données

| Transformation | Paramètres |
|----------------|------------|
| Resize | 224×224 |
| RandomHorizontalFlip | $p=0.5$ |
| RandomRotation | $\pm 15°$ |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 |

### 4.3 Métriques d'Évaluation

**Accuracy :**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision et Recall :**

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score :**

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 5. Pipeline de Réentraînement

### 5.1 Déclenchement Automatique

Le watcher détecte les nouvelles versions de dataset selon l'algorithme :

```
DÉBUT
│
├── Charger état précédent (last_processed_id)
│
├── Récupérer dernier dataset ClearML
│
├── SI nouveau_id ≠ last_processed_id ALORS
│   ├── Déclencher pipeline(nouveau_id)
│   └── Sauvegarder état(nouveau_id)
│
└── FIN
```

### 5.2 Configuration ClearML

```python
api {
    web_server: https://app.clear.ml/
    api_server: https://api.clear.ml
    files_server: https://files.clear.ml
    credentials {
        access_key = "..."
        secret_key = "..."
    }
}
```

### 5.3 Décorateurs Pipeline

```python
@PipelineDecorator.component(cache=True)
def prepare_data(dataset_id: str) -> dict:
    ...

@PipelineDecorator.component(cache=False)
def train_model(data_info: dict, epochs: int) -> dict:
    ...

@PipelineDecorator.pipeline(name="AutoRetrain", project="MLOps_CatsVsDogs")
def retrain_pipeline(dataset_id: str, epochs: int = 10):
    data_info = prepare_data(dataset_id)
    train_result = train_model(data_info, epochs)
    register_model(train_result, dataset_id)
```

---

## 6. Résultats et Métriques

### 6.1 Performance du Modèle Baseline

| Métrique | Valeur |
|----------|--------|
| Validation Accuracy | ~95% |
| Training Loss | ~0.15 |
| Validation Loss | ~0.18 |
| Temps d'entraînement | ~30 min (GPU) |

### 6.2 Courbes d'Apprentissage

**Évolution de la loss :**

$$\mathcal{L}(t) = \mathcal{L}_0 \cdot e^{-\lambda t} + \mathcal{L}_{\infty}$$

Où :
- $\mathcal{L}_0$ = loss initiale
- $\lambda$ = taux de convergence
- $\mathcal{L}_{\infty}$ = loss asymptotique

### 6.3 Analyse du Feedback

La boucle de feedback améliore itérativement le modèle :

$$\text{Accuracy}_{n+1} = \text{Accuracy}_n + \Delta_{\text{feedback}}$$

Où $\Delta_{\text{feedback}}$ dépend de la qualité et quantité des corrections.

---

## 7. Conclusion

### 7.1 Réalisations

Ce projet démontre une implémentation complète des pratiques MLOps :

✅ **Tracking des expériences** avec ClearML  
✅ **Versioning des données** et des modèles  
✅ **Pipeline automatisée** de réentraînement  
✅ **Boucle de feedback** utilisateur  
✅ **Model Registry** centralisé  

### 7.2 Avantages de l'Approche

| Aspect | Bénéfice |
|--------|----------|
| **Reproductibilité** | Chaque expérience est traçable |
| **Automatisation** | Réentraînement sans intervention |
| **Collaboration** | Interface ClearML partagée |
| **Amélioration continue** | Feedback → Données → Meilleur modèle |

### 7.3 Perspectives

- Déploiement sur **Kubernetes** avec ClearML Serving
- Ajout de **tests automatisés** (data validation, model validation)
- Intégration de **monitoring** en production
- Extension à la **classification multi-classes**

---

## Annexes

### A. Commandes Principales

```bash
# Installation
pip install -r requirements.txt

# Téléchargement dataset
python src/download_dataset.py

# Entraînement
python src/train_baseline.py --data-path data/cats_vs_dogs --epochs 10

# Interface Gradio
python src/gradio_app.py

# Versioning dataset
python src/dataset_versioning.py --add-feedback

# Watcher
python src/watcher_trigger.py --watch
```

### B. Références

1. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
2. Kingma, D. P., & Ba, J. "Adam: A Method for Stochastic Optimization." ICLR 2015.
3. ClearML Documentation: https://clear.ml/docs/

---

*Rapport généré le 4 Décembre 2024*
