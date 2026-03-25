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
import numpy as np
import joblib

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_model_from_pkl(pkl_path=None):
    """Load model from .pkl file."""
    if pkl_path is None:
        pkl_path = os.path.join(OUTPUT_DIR, 'plant_disease_model.pkl')
    
    if not os.path.exists(pkl_path):
        print(f"[ERROR] Model not found: {pkl_path}")
        print(f"[INFO] Run 'py -3.13 train_model.py' first.")
        sys.exit(1)
    
    print(f"[INFO] Loading model from: {pkl_path}")
    model_package = joblib.load(pkl_path)
    
    # Rebuild model architecture
    num_classes = model_package['num_classes']
    model = models.mobilenet_v2(weights=None)
    
    import torch.nn as nn
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
    
    print(f"[INFO] Model: {model_package['model_name']}")
    print(f"[INFO] Classes: {model_package['num_classes']}")
    print(f"[INFO] Accuracy: {model_package['accuracy']*100:.2f}%")
    
    return model, model_package


def preprocess_image(img_path, img_size=224):
    """Load and preprocess image."""
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def predict_disease(model, model_package, img_path, top_k=5):
    """Predict plant disease."""
    img_size = model_package.get('img_size', 224)
    idx_to_class = model_package.get('idx_to_class', {})
    
    img_tensor = preprocess_image(img_path, img_size)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    top_probs, top_indices = probabilities.topk(top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = idx_to_class.get(str(idx.item()), idx_to_class.get(idx.item(), f"Class_{idx.item()}"))
        results.append({
            'class_index': idx.item(),
            'class_name': class_name,
            'confidence': prob.item(),
            'confidence_pct': f"{prob.item() * 100:.2f}%"
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="PlantVillage Disease Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--model", type=str, default=None, help="Path to .pkl model file")
    parser.add_argument("--top_k", type=int, default=5, help="Top predictions (default: 5)")
    args = parser.parse_args()
    
    model, model_package = load_model_from_pkl(args.model)
    
    print(f"\n[INFO] Predicting disease for: {args.image}")
    results = predict_disease(model, model_package, args.image, args.top_k)
    
    print(f"\n{'='*60}")
    print(f"  🌿 PREDICTION RESULTS")
    print(f"{'='*60}")
    
    for i, result in enumerate(results):
        bar_len = int(result['confidence'] * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        marker = " ◄── PREDICTED" if i == 0 else ""
        print(f"  [{i+1}] {result['class_name']}")
        print(f"      {bar} {result['confidence_pct']}{marker}")
    
    print(f"\n{'='*60}")
    
    top = results[0]
    parts = top['class_name'].split('___')
    if len(parts) == 2:
        plant, disease = parts
        plant = plant.replace('_', ' ')
        disease = disease.replace('_', ' ')
        print(f"  🌱 Plant: {plant}")
        if 'healthy' in disease.lower():
            print(f"  ✅ Status: HEALTHY")
        else:
            print(f"  ⚠️  Disease: {disease}")
        print(f"  📊 Confidence: {top['confidence_pct']}")
    
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    main()
