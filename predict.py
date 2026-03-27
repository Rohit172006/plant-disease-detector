"""
PlantVillage Disease Prediction Script (PyTorch)
==================================================
Load the trained .pkl model and predict plant diseases from images.

Usage:
    py -3.13 predict.py --image path/to/leaf_image.jpg
    py -3.13 predict.py --image path/to/leaf_image.jpg --top_k 5
"""

import os
import sys
import argparse
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ============================================================
# Load Model
# ============================================================
def load_model_from_pkl(pkl_path=None):
    """Load trained model from .pkl file"""
    if pkl_path is None:
        pkl_path = os.path.join(OUTPUT_DIR, 'plant_disease_model.pkl')

    if not os.path.exists(pkl_path):
        print(f"[ERROR] Model not found: {pkl_path}")
        sys.exit(1)

    print(f"[INFO] Loading model from: {pkl_path}")
    model_package = joblib.load(pkl_path)

    num_classes = model_package['num_classes']

    # Rebuild MobileNetV2
    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    # Load weights
    model.load_state_dict(model_package['model_state_dict'])
    model.eval()

    print(f"[INFO] Model Loaded Successfully")
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Accuracy: {model_package['accuracy']*100:.2f}%")

    return model, model_package


# ============================================================
# Preprocess Image
# ============================================================
def preprocess_image(img_path, img_size=224):
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    return image


# ============================================================
# Predict Function
# ============================================================
def predict_disease(model, model_package, img_path, top_k=3):
    img_size = model_package.get('img_size', 224)

    # FIX: normalize keys properly
    idx_to_class = {
        int(k): v for k, v in model_package.get('idx_to_class', {}).items()
    }

    image = preprocess_image(img_path, img_size)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)

    results = []

    for prob, idx in zip(top_probs, top_indices):
        class_name = idx_to_class.get(idx.item(), f"Class_{idx.item()}")

        # Clean display name
        formatted_name = class_name.replace("___", " - ").replace("_", " ")

        results.append({
            "class_index": idx.item(),
            "class_name": class_name,
            "formatted_name": formatted_name,
            "confidence": float(prob.item()),
            "confidence_pct": f"{prob.item()*100:.2f}%"
        })

    return results


# ============================================================
# Pretty Output (CLI)
# ============================================================
def print_results(results):
    print("\n" + "="*60)
    print("  🌿 PREDICTION RESULTS")
    print("="*60)

    for i, result in enumerate(results):
        bar_len = int(result['confidence'] * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        marker = " ◄── PREDICTED" if i == 0 else ""

        print(f"  [{i+1}] {result['formatted_name']}")
        print(f"      {bar} {result['confidence_pct']}{marker}")

    print("="*60)

    # Top result breakdown
    top = results[0]
    parts = top['class_name'].split('___')

    plant = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else "Unknown"

    print(f"  🌱 Plant: {plant}")

    if "healthy" in disease.lower():
        print(f"  ✅ Status: HEALTHY")
    else:
        print(f"  ⚠️ Disease: {disease}")

    print(f"  📊 Confidence: {top['confidence_pct']}")

    # Low confidence warning
    if top['confidence'] < 0.6:
        print("  ⚠️ Low confidence. Try a clearer image.")

    print("="*60 + "\n")


# ============================================================
# CLI Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Plant Disease Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model", type=str, default=None, help="Path to .pkl model")
    parser.add_argument("--top_k", type=int, default=3, help="Top predictions")

    args = parser.parse_args()

    model, model_package = load_model_from_pkl(args.model)

    print(f"\n[INFO] Predicting: {args.image}")
    results = predict_disease(model, model_package, args.image, args.top_k)

    print_results(results)


# ============================================================
# For Flask / Import Use
# ============================================================
def get_prediction(image_path):
    model, model_package = load_model_from_pkl()
    results = predict_disease(model, model_package, image_path, top_k=1)
    return results[0]


if __name__ == "__main__":
    main()