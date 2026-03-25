"""
PlantVillage CNN Model Training Pipeline (PyTorch)
====================================================
Trains a CNN model (MobileNetV2 Transfer Learning) for plant disease classification.
Outputs: plant_disease_model.pkl, class_indices.json, training_history.png

Dataset: PlantVillage Dataset (https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
Architecture: MobileNetV2 (Transfer Learning) via PyTorch/torchvision
Classes: 38 plant disease categories

Usage:
    py -3.13 train_model.py
    py -3.13 train_model.py --epochs 20 --batch_size 32 --img_size 224
"""

import os
import sys
import json
import argparse
import warnings
import time
import copy
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PlantVillage CNN Model (PyTorch)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Fine-tuning epochs (default: 5)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default: 2)")
    return parser.parse_args()


def find_dataset_directory(base_path):
    """
    Auto-detect the correct dataset directory.
    PlantVillage dataset has 3 folders: color, grayscale, segmented.
    We use the 'color' folder for best accuracy.
    """
    possible_paths = [
        os.path.join(base_path, "plantvillage dataset", "color"),
        os.path.join(base_path, "plantvillage dataset", "Color"),
        os.path.join(base_path, "PlantVillage", "color"),
        os.path.join(base_path, "color"),
        os.path.join(base_path, "plantvillage-dataset", "color"),
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            return p
    
    # Try to find any directory with plant disease class folders
    for root, dirs, files in os.walk(base_path):
        plant_keywords = ['apple', 'tomato', 'grape', 'corn', 'potato', 'pepper']
        matching = [d for d in dirs if any(k in d.lower() for k in plant_keywords)]
        if len(matching) >= 3:
            return root
    
    return base_path


def create_data_loaders(data_dir, img_size, batch_size, num_workers):
    """
    Create train and validation DataLoaders with augmentation.
    80/20 train-validation split.
    """
    print(f"\n{'='*60}")
    print(f"  DATA LOADING")
    print(f"{'='*60}")
    print(f"  Dataset path: {data_dir}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset for splitting
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_names)
    total_images = len(full_dataset)
    
    print(f"  Number of classes: {num_classes}")
    print(f"  Total images: {total_images}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Batch size: {batch_size}")
    
    # 80/20 split
    train_size = int(0.8 * total_images)
    val_size = total_images - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Apply transforms
    train_dataset.dataset = copy.copy(full_dataset)
    # We need wrapper datasets to apply different transforms
    train_data = TransformSubset(train_dataset, train_transform)
    val_data = TransformSubset(val_dataset, val_transform)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Print class names
    print(f"\n  Class names:")
    for idx, name in enumerate(class_names):
        print(f"    [{idx:2d}] {name}")
    
    return train_loader, val_loader, class_names, class_to_idx


class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply transforms to a Subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # img is a PIL Image from ImageFolder
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes, device):
    """
    Build CNN model using MobileNetV2 transfer learning.
    
    Architecture:
    - MobileNetV2 backbone (ImageNet pretrained, initially frozen)
    - Custom classifier head with dropout
    """
    print(f"\n{'='*60}")
    print(f"  MODEL ARCHITECTURE")
    print(f"{'='*60}")
    
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze all base layers initially
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier head
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
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Base model: MobileNetV2 (ImageNet weights)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Output classes: {num_classes}")
    print(f"  Device: {device}")
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch, total_epochs):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]  ", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_classes, device, 
                epochs, fine_tune_epochs, learning_rate):
    """
    Two-phase training:
    Phase 1: Train classifier head (backbone frozen)
    Phase 2: Fine-tune top layers of backbone
    """
    criterion = nn.CrossEntropyLoss()
    history = defaultdict(list)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # ===== PHASE 1: Train classifier head =====
    print(f"\n{'='*60}")
    print(f"  PHASE 1: TRAINING CLASSIFIER HEAD")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")
    
    # Only optimize classifier parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=1e-4
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        start = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, epochs
        )
        
        scheduler.step(val_acc)
        elapsed = time.time() - start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"Time: {elapsed:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"    ✓ New best model saved! (Acc: {best_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered (patience={patience})")
                break
    
    # ===== PHASE 2: Fine-tune backbone =====
    if fine_tune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: FINE-TUNING BACKBONE")
        print(f"{'='*60}")
        print(f"  Epochs: {fine_tune_epochs}")
        
        # Unfreeze last ~30% of backbone layers
        all_layers = list(model.features.children())
        unfreeze_from = max(0, len(all_layers) - 5)  # Unfreeze last 5 blocks
        
        for i, layer in enumerate(all_layers):
            if i >= unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Unfrozen layers from block: {unfreeze_from}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Lower learning rate for fine-tuning
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate / 10, weight_decay=1e-4
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        patience_counter = 0
        total_epochs_so_far = len(history['train_loss'])
        
        for epoch in range(fine_tune_epochs):
            start = time.time()
            global_epoch = total_epochs_so_far + epoch
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, 
                global_epoch, total_epochs_so_far + fine_tune_epochs
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, device,
                global_epoch, total_epochs_so_far + fine_tune_epochs
            )
            
            scheduler.step(val_acc)
            elapsed = time.time() - start
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"  Epoch {global_epoch+1:3d}/{total_epochs_so_far+fine_tune_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
                  f"Time: {elapsed:.1f}s")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
                print(f"    ✓ New best model saved! (Acc: {best_acc*100:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"    Early stopping triggered (patience=3)")
                    break
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"\n  Best validation accuracy: {best_acc*100:.2f}%")
    
    return model, dict(history), best_acc


def evaluate_model(model, val_loader, class_names, device):
    """Evaluate model and generate classification report."""
    print(f"\n{'='*60}")
    print(f"  MODEL EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = running_loss / total
    
    print(f"  Validation Loss: {avg_loss:.4f}")
    print(f"  Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return avg_loss, accuracy, all_preds, all_labels, report


def plot_training_history(history, output_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_range = range(1, len(history['train_acc']) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs_range, history['train_acc'], label='Train Accuracy', linewidth=2, color='#2196F3')
    axes[0].plot(epochs_range, history['val_acc'], label='Val Accuracy', linewidth=2, color='#FF5722')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Loss plot
    axes[1].plot(epochs_range, history['train_loss'], label='Train Loss', linewidth=2, color='#2196F3')
    axes[1].plot(epochs_range, history['val_loss'], label='Val Loss', linewidth=2, color='#FF5722')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Training history plot: {save_path}")


def plot_confusion_matrix(true_classes, predicted_classes, class_names, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(true_classes, predicted_classes)
    
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Confusion matrix: {save_path}")


def save_model_as_pkl(model, class_to_idx, class_names, accuracy, output_dir, img_size=224):
    """
    Save the trained model as a .pkl file using joblib.
    
    The pkl file contains:
    - model_state_dict: PyTorch model weights
    - model_architecture: Description of architecture
    - class_to_idx: Mapping of class names to indices
    - class_names: List of class names
    - img_size: Input image size
    - accuracy: Validation accuracy
    - preprocessing: Normalization parameters
    """
    print(f"\n{'='*60}")
    print(f"  SAVING MODEL")
    print(f"{'='*60}")
    
    # Save PyTorch model
    pth_path = os.path.join(output_dir, 'plant_disease_model.pth')
    torch.save(model.state_dict(), pth_path)
    print(f"  [SAVED] PyTorch model weights: {pth_path}")
    
    # Save complete model (architecture + weights)
    full_model_path = os.path.join(output_dir, 'plant_disease_model_full.pth')
    torch.save(model, full_model_path)
    print(f"  [SAVED] Full PyTorch model: {full_model_path}")
    
    # Create comprehensive pkl package
    model_package = {
        'model_name': 'PlantDisease_MobileNetV2',
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'class_names': class_names,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'num_classes': len(class_names),
        'img_size': img_size,
        'accuracy': float(accuracy),
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'target_size': (img_size, img_size),
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
    
    pkl_path = os.path.join(output_dir, 'plant_disease_model.pkl')
    joblib.dump(model_package, pkl_path)
    print(f"  [SAVED] PKL model package: {pkl_path}")
    
    # Save class indices separately
    class_path = os.path.join(output_dir, 'class_indices.json')
    with open(class_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"  [SAVED] Class indices: {class_path}")
    
    # Save inverted class indices
    inv_class_path = os.path.join(output_dir, 'class_names.json')
    with open(inv_class_path, 'w') as f:
        json.dump({str(v): k for k, v in class_to_idx.items()}, f, indent=2)
    print(f"  [SAVED] Class names: {inv_class_path}")
    
    return pkl_path


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"  🌿 PLANTVILLAGE DISEASE CLASSIFICATION MODEL")
    print(f"  CNN Training Pipeline (MobileNetV2 - PyTorch)")
    print(f"{'='*60}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print(f"  [INFO] No GPU detected. Training on CPU (will be slower).")
    
    # Find dataset
    data_dir = args.data_dir or DATASET_DIR
    data_dir = find_dataset_directory(data_dir)
    
    if not os.path.exists(data_dir):
        print(f"\n  [ERROR] Dataset not found at: {data_dir}")
        print(f"  Please run 'py -3.13 download_dataset.py' first.")
        print(f"  Or specify path: py -3.13 train_model.py --data_dir <path>")
        sys.exit(1)
    
    # Verify dataset
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(subdirs) < 2:
        print(f"\n  [ERROR] Not enough class folders in: {data_dir}")
        print(f"  Found: {subdirs}")
        sys.exit(1)
    
    # Create data loaders
    train_loader, val_loader, class_names, class_to_idx = create_data_loaders(
        data_dir, args.img_size, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)
    
    # Build model
    model = build_model(num_classes, device)
    
    # Train model
    model, history, best_acc = train_model(
        model, train_loader, val_loader, num_classes, device,
        args.epochs, args.fine_tune_epochs, args.learning_rate
    )
    
    # Evaluate
    loss, accuracy, pred_classes, true_classes, report = evaluate_model(
        model, val_loader, class_names, device
    )
    
    # Plot training history
    plot_training_history(history, OUTPUT_DIR)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_classes, pred_classes, class_names, OUTPUT_DIR)
    
    # Save model as PKL
    pkl_path = save_model_as_pkl(
        model, class_to_idx, class_names, accuracy, OUTPUT_DIR, args.img_size
    )
    
    # Save evaluation report
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
    eval_report = {
        'validation_loss': float(loss),
        'validation_accuracy': float(accuracy),
        'num_classes': num_classes,
        'class_names': class_names,
        'classification_report': report
    }
    with open(report_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    print(f"  [SAVED] Evaluation report: {report_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"  🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Validation Accuracy: {accuracy*100:.2f}%")
    print(f"  Validation Loss: {loss:.4f}")
    print(f"  Number of Classes: {num_classes}")
    print(f"\n  Output files saved to: {OUTPUT_DIR}")
    print(f"  ├── plant_disease_model.pkl      (PKL model package)")
    print(f"  ├── plant_disease_model.pth       (PyTorch weights)")
    print(f"  ├── plant_disease_model_full.pth  (Full model)")
    print(f"  ├── best_model.pth               (Best checkpoint)")
    print(f"  ├── class_indices.json            (Class mappings)")
    print(f"  ├── class_names.json              (Index to name)")
    print(f"  ├── training_history.png          (Training curves)")
    print(f"  ├── confusion_matrix.png          (Confusion matrix)")
    print(f"  └── evaluation_report.json        (Metrics report)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
