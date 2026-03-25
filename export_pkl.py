"""
Export saved best_model.pth checkpoint to .pkl file.
Generates all final output files without re-training.
"""
import os
import sys
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def find_data_dir(base):
    paths = [
        os.path.join(base, "plantvillage dataset", "color"),
        os.path.join(base, "plantvillage dataset", "Color"),
        os.path.join(base, "color"),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    for root, dirs, _ in os.walk(base):
        kw = ['apple', 'tomato', 'grape', 'corn', 'potato', 'pepper']
        if len([d for d in dirs if any(k in d.lower() for k in kw)]) >= 3:
            return root
    return base

def main():
    device = torch.device('cpu')
    
    # Find dataset to get class names
    data_dir = find_data_dir(DATASET_DIR)
    print(f"[INFO] Dataset: {data_dir}")
    
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_names)
    print(f"[INFO] Classes: {num_classes}")
    
    # Build model architecture
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
    
    # Load best checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print("[INFO] Model loaded successfully!")
    
    # Evaluate on validation set
    print("[INFO] Running evaluation...")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    
    # Use 20% as validation (same split seed)
    total = len(val_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    _, val_subset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = correct / total_samples
    print(f"[INFO] Validation Accuracy: {accuracy*100:.2f}%")
    
    # Save PKL
    model_package = {
        'model_name': 'PlantDisease_MobileNetV2',
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'class_names': class_names,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'num_classes': num_classes,
        'img_size': 224,
        'accuracy': float(accuracy),
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'target_size': (224, 224),
            'color_mode': 'rgb'
        },
        'training_info': {
            'base_model': 'MobileNetV2',
            'pretrained_on': 'ImageNet',
            'fine_tuned': True,
            'framework': 'PyTorch',
            'torch_version': torch.__version__
        }
    }
    
    pkl_path = os.path.join(OUTPUT_DIR, 'plant_disease_model.pkl')
    joblib.dump(model_package, pkl_path)
    print(f"[SAVED] PKL: {pkl_path}")
    
    # Save .pth weights
    pth_path = os.path.join(OUTPUT_DIR, 'plant_disease_model.pth')
    torch.save(model.state_dict(), pth_path)
    print(f"[SAVED] PTH: {pth_path}")
    
    # Save full model
    full_path = os.path.join(OUTPUT_DIR, 'plant_disease_model_full.pth')
    torch.save(model, full_path)
    print(f"[SAVED] Full model: {full_path}")
    
    # Save class indices
    with open(os.path.join(OUTPUT_DIR, 'class_indices.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'class_names.json'), 'w') as f:
        json.dump({str(v): k for k, v in class_to_idx.items()}, f, indent=2)
    print("[SAVED] Class indices and names JSON")
    
    # Classification report
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save eval report
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump({
            'validation_accuracy': float(accuracy),
            'num_classes': num_classes,
            'class_names': class_names,
            'classification_report': report
        }, f, indent=2)
    print("[SAVED] Evaluation report")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("[SAVED] Confusion matrix plot")
    
    print(f"\n{'='*60}")
    print(f"  🎉 EXPORT COMPLETE!")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Classes: {num_classes}")
    print(f"  PKL file: {pkl_path}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
