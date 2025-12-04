"""
gradio_app.py - Interface utilisateur Gradio pour la classification d'images

Ce script :
1. Charge le modèle depuis un chemin local ou le Model Registry ClearML
2. Crée une interface Gradio pour la prédiction
3. Collecte le feedback utilisateur et sauvegarde les images annotées
4. Log les prédictions et feedbacks dans ClearML

Usage:
    python src/gradio_app.py
    
    # Avec un modèle spécifique:
    python src/gradio_app.py --model-path models/best_model.pth
    
    # Avec le Model Registry ClearML:
    python src/gradio_app.py --use-registry
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import torch
import gradio as gr
from PIL import Image

from clearml import Task, InputModel

from utils import (
    load_model,
    create_model,
    preprocess_image,
    get_latest_model_path,
    save_feedback_image,
    get_total_feedback_count,
    count_feedback_images,
    get_device,
    setup_clearml_credentials,
    CLEARML_PROJECT_NAME,
    MODELS_DIR,
    FEEDBACK_DIR,
    DEFAULT_CLASSES
)


# =============================================================================
# VARIABLES GLOBALES
# =============================================================================

# Ces variables seront initialisées au démarrage
MODEL = None
CLASSES = DEFAULT_CLASSES
DEVICE = "cpu"
TASK = None


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Application Gradio pour la classification d'images")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Chemin vers le modèle local (.pth)"
    )
    parser.add_argument(
        "--use-registry", action="store_true",
        help="Charger le modèle depuis le Model Registry ClearML"
    )
    parser.add_argument(
        "--model-name", type=str, default="baseline_model_resnet18",
        help="Nom du modèle dans le Registry (si --use-registry)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Créer un lien public Gradio"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port pour l'interface web (default: 7860)"
    )
    return parser.parse_args()


def load_model_from_registry(model_name: str) -> Tuple[torch.nn.Module, list]:
    """
    Charge un modèle depuis le ClearML Model Registry.
    
    Returns:
        Tuple (model, classes)
    """
    print(f"Chargement du modèle '{model_name}' depuis le Model Registry...")
    
    input_model = InputModel(
        project=CLEARML_PROJECT_NAME,
        name=model_name
    )
    
    # Télécharger les poids
    model_path = input_model.get_weights()
    
    # Récupérer les métadonnées
    design = input_model.config_dict
    classes = design.get("classes", DEFAULT_CLASSES)
    architecture = design.get("architecture", "resnet18")
    num_classes = len(classes)
    
    # Charger le modèle
    model = create_model(
        num_classes=num_classes,
        pretrained=False,
        model_name=architecture
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print(f"✓ Modèle chargé: {architecture}, {num_classes} classes")
    return model, classes


def initialize_model(args) -> Tuple[torch.nn.Module, list]:
    """
    Initialise le modèle selon les arguments.
    
    Returns:
        Tuple (model, classes)
    """
    global DEVICE
    DEVICE = get_device()
    print(f"Device: {DEVICE}")
    
    if args.use_registry:
        return load_model_from_registry(args.model_name)
    
    elif args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        model, metadata = load_model(model_path, device=DEVICE)
        classes = metadata.get("classes", DEFAULT_CLASSES) if metadata else DEFAULT_CLASSES
        return model, classes
    
    else:
        # Chercher le dernier modèle local
        model_path = get_latest_model_path()
        if model_path:
            print(f"Utilisation du dernier modèle local: {model_path}")
            model, metadata = load_model(model_path, device=DEVICE)
            # Valider les classes - ignorer si ce sont des noms de dossiers incorrects
            if metadata and "classes" in metadata:
                detected_classes = metadata["classes"]
                # Vérifier si les classes sont valides (pas train/val)
                if "train" in detected_classes or "val" in detected_classes:
                    print(f"⚠️  Classes invalides détectées: {detected_classes}")
                    print(f"    Utilisation des classes par défaut: {DEFAULT_CLASSES}")
                    classes = DEFAULT_CLASSES
                else:
                    classes = detected_classes
            else:
                classes = DEFAULT_CLASSES
            return model, classes
        else:
            # Mode démo sans modèle réel
            print("⚠️  Aucun modèle trouvé. Mode démonstration...")
            model = create_model(num_classes=len(DEFAULT_CLASSES), pretrained=True)
            model.to(DEVICE)
            model.eval()
            return model, DEFAULT_CLASSES


# =============================================================================
# FONCTIONS DE PRÉDICTION ET FEEDBACK
# =============================================================================

def predict(image: Image.Image) -> str:
    """
    Effectue une prédiction sur une image.
    
    Args:
        image: Image PIL
        
    Returns:
        String avec les prédictions et probabilités
    """
    global MODEL, CLASSES, DEVICE, TASK
    
    if image is None:
        return "Veuillez fournir une image."
    
    try:
        # Préprocesser l'image
        input_tensor = preprocess_image(image, for_training=False)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        
        # Prédiction
        with torch.no_grad():
            outputs = MODEL(input_batch)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Formater les résultats
        results = []
        sorted_indices = torch.argsort(probabilities, descending=True)
        
        for idx in sorted_indices[:3]:  # Top 3
            class_name = CLASSES[idx]
            prob = probabilities[idx].item() * 100
            results.append(f"• {class_name}: {prob:.1f}%")
        
        predicted_class = CLASSES[sorted_indices[0]]
        prediction_text = f"**Prédiction: {predicted_class}**\n\n" + "\n".join(results)
        
        # Log dans ClearML si une tâche est active
        if TASK:
            TASK.get_logger().report_text(
                f"Prediction: {predicted_class}",
                print_console=False
            )
        
        return prediction_text
        
    except Exception as e:
        return f"Erreur de prédiction: {str(e)}"


def get_predicted_class(image: Image.Image) -> str:
    """Retourne uniquement la classe prédite."""
    global MODEL, CLASSES, DEVICE
    
    if image is None:
        return ""
    
    try:
        input_tensor = preprocess_image(image, for_training=False)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = MODEL(input_batch)
            _, predicted = outputs.max(1)
        
        return CLASSES[predicted.item()]
    except:
        return ""


def handle_feedback(
    image: Image.Image,
    true_label: str,
    prediction_display: str
) -> str:
    """
    Enregistre le feedback utilisateur.
    
    Args:
        image: Image PIL
        true_label: Label correct fourni par l'utilisateur
        prediction_display: Affichage de la prédiction (pour extraire la classe prédite)
        
    Returns:
        Message de confirmation
    """
    global TASK
    
    if image is None:
        return "❌ Veuillez d'abord charger une image."
    
    if not true_label or true_label.strip() == "":
        return "❌ Veuillez sélectionner ou saisir le label correct."
    
    true_label = true_label.strip().lower().replace(" ", "_")
    
    # Récupérer la prédiction
    predicted_label = get_predicted_class(image)
    if not predicted_label:
        predicted_label = "unknown"
    
    try:
        # Sauvegarder l'image
        saved_path = save_feedback_image(image, true_label, predicted_label)
        
        # Log dans ClearML
        if TASK:
            is_correct = true_label == predicted_label
            TASK.get_logger().report_single_value(
                name="feedback_count",
                value=get_total_feedback_count()
            )
            TASK.get_logger().report_text(
                f"Feedback: predicted={predicted_label}, true={true_label}, correct={is_correct}",
                print_console=False
            )
        
        # Statistiques
        stats = count_feedback_images()
        total = get_total_feedback_count()
        
        return f"""✅ **Feedback enregistré !**

• Image sauvegardée: `{saved_path.name}`
• Label: **{true_label}**
• Prédiction originale: {predicted_label}
• Total feedbacks collectés: **{total}**

_Cette image sera utilisée pour améliorer le modèle lors du prochain réentraînement._"""
        
    except Exception as e:
        return f"❌ Erreur lors de la sauvegarde: {str(e)}"


def get_feedback_stats() -> str:
    """Retourne les statistiques de feedback."""
    stats = count_feedback_images()
    total = get_total_feedback_count()
    
    if not stats:
        return "Aucun feedback collecté pour le moment."
    
    lines = [f"**Total: {total} images**\n"]
    for class_name, count in sorted(stats.items()):
        lines.append(f"• {class_name}: {count}")
    
    return "\n".join(lines)


# =============================================================================
# INTERFACE GRADIO
# =============================================================================

def create_interface(classes: List[str], share: bool = False, port: int = 7860):
    """Crée et lance l'interface Gradio."""
    
    with gr.Blocks(
        title="Classification d'Images - MLOps"
    ) as demo:
        
        gr.Markdown("""
# 🖼️ Classification d'Images - MLOps Demo

Uploadez une image pour obtenir une prédiction. Si la prédiction est incorrecte, 
vous pouvez fournir le label correct pour améliorer le modèle.
        """)
        
        with gr.Row():
            # Colonne gauche : Image et prédiction
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Image à classifier")
                image_input = gr.Image(
                    type="pil",
                    label="Charger une image",
                    sources=["upload", "clipboard", "webcam"]
                )
                predict_btn = gr.Button("🔍 Prédire", variant="primary")
                
                gr.Markdown("### 📊 Résultat")
                prediction_output = gr.Markdown(
                    value="_Chargez une image et cliquez sur 'Prédire'_"
                )
            
            # Colonne droite : Feedback
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Feedback (Correction)")
                gr.Markdown(
                    "_Si la prédiction est incorrecte, sélectionnez le bon label:_"
                )
                
                label_dropdown = gr.Dropdown(
                    choices=classes,
                    label="Label correct",
                    info="Sélectionnez dans la liste ou tapez un nouveau label"
                )
                
                label_text = gr.Textbox(
                    label="Ou saisissez un nouveau label",
                    placeholder="ex: chat_persan"
                )
                
                feedback_btn = gr.Button("✅ Envoyer le feedback", variant="secondary")
                
                feedback_output = gr.Markdown(value="")
                
                gr.Markdown("---")
                gr.Markdown("### 📈 Statistiques de feedback")
                stats_output = gr.Markdown(value=get_feedback_stats())
                refresh_stats_btn = gr.Button("🔄 Rafraîchir")
        
        gr.Markdown("""
---
### ℹ️ Comment ça marche

1. **Uploadez** une image via drag & drop, presse-papier ou webcam
2. **Cliquez** sur "Prédire" pour obtenir la classification
3. **Si la prédiction est incorrecte**, sélectionnez ou saisissez le bon label
4. **Envoyez** votre feedback - l'image sera sauvegardée pour le réentraînement

_Les feedbacks sont collectés et utilisés pour créer de nouvelles versions du dataset, 
ce qui déclenche automatiquement un réentraînement du modèle via ClearML Pipeline._
        """)
        
        # Événements
        predict_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[prediction_output]
        )
        
        # Prédiction automatique au chargement d'image
        image_input.change(
            fn=predict,
            inputs=[image_input],
            outputs=[prediction_output]
        )
        
        def submit_feedback(image, dropdown_label, text_label, prediction_display):
            # Priorité au dropdown, sinon au texte
            label = dropdown_label if dropdown_label else text_label
            return handle_feedback(image, label, prediction_display)
        
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[image_input, label_dropdown, label_text, prediction_output],
            outputs=[feedback_output]
        )
        
        refresh_stats_btn.click(
            fn=get_feedback_stats,
            inputs=[],
            outputs=[stats_output]
        )
    
    return demo


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Point d'entrée principal."""
    global MODEL, CLASSES, TASK
    
    args = parse_args()
    
    # Configuration ClearML
    setup_clearml_credentials()
    
    # Initialiser la tâche ClearML pour tracker les prédictions
    TASK = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="gradio_inference_app",
        task_type=Task.TaskTypes.inference
    )
    
    print("=" * 60)
    print("APPLICATION GRADIO - CLASSIFICATION D'IMAGES")
    print("=" * 60)
    
    # Charger le modèle
    MODEL, CLASSES = initialize_model(args)
    
    print(f"Classes: {CLASSES}")
    print(f"Feedback directory: {FEEDBACK_DIR}")
    print(f"Feedbacks existants: {get_total_feedback_count()}")
    print("=" * 60)
    
    # Créer et lancer l'interface
    demo = create_interface(
        classes=CLASSES,
        share=args.share,
        port=args.port
    )
    
    print(f"\n🚀 Lancement de l'interface sur http://localhost:{args.port}")
    if args.share:
        print("📡 Un lien public sera généré...")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
