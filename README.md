# 🌿 PlantVillage Plant Disease Classification

CNN-based plant disease classifier using **MobileNetV2 Transfer Learning** with **PyTorch**.

Trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) containing 50,000+ images across **38 disease classes**.

---

## 📁 Project Structure

```
plants/
├── download_dataset.py    # Download dataset from Kaggle
├── train_model.py         # Train CNN model (main script)
├── predict.py             # Predict disease from leaf image
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── dataset/               # Downloaded dataset (auto-created)
│   └── plantvillage dataset/
│       └── color/         # RGB leaf images (38 classes)
└── output/                # Training outputs (auto-created)
    ├── plant_disease_model.pkl      # ← YOUR PKL FILE
    ├── plant_disease_model.pth      # PyTorch weights
    ├── plant_disease_model_full.pth # Complete model
    ├── best_model.pth              # Best checkpoint
    ├── class_indices.json          # Class name → index
    ├── class_names.json            # Index → class name
    ├── training_history.png        # Accuracy/Loss curves
    ├── confusion_matrix.png        # Confusion matrix
    └── evaluation_report.json      # Detailed metrics
```

---

## 🚀 Quick Start

### Step 1: Set up Kaggle API

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to **API** section → Click **"Create New Token"**
3. This downloads `kaggle.json`
4. Place it in: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### Step 2: Download Dataset

```bash
py -3.13 download_dataset.py
```

**OR** manually download from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset):
1. Download and extract to `plants/dataset/`
2. Ensure the color images are at: `plants/dataset/plantvillage dataset/color/`

### Step 3: Train the Model

```bash
py -3.13 train_model.py
```

With custom parameters:
```bash
py -3.13 train_model.py --epochs 20 --batch_size 64 --learning_rate 0.001 --fine_tune_epochs 10
```

### Step 4: Predict Diseases

```bash
py -3.13 predict.py --image path/to/leaf_image.jpg
```

---

## 🏗️ Model Architecture

| Component | Detail |
|-----------|--------|
| **Backbone** | MobileNetV2 (ImageNet pretrained) |
| **Classifier** | Dense(512) → BN → Dropout(0.4) → Dense(256) → BN → Dropout(0.3) → Dense(38) |
| **Training** | 2-phase: frozen backbone → fine-tune last 5 blocks |
| **Augmentation** | Rotation, flip, color jitter, random crop, affine |
| **Optimizer** | Adam (lr=0.001 → 0.0001 for fine-tuning) |
| **Output** | 38 classes (plant + disease combinations) |

---

## 🏷️ Disease Classes (38)

| # | Class Name |
|---|-----------|
| 0 | Apple___Apple_scab |
| 1 | Apple___Black_rot |
| 2 | Apple___Cedar_apple_rust |
| 3 | Apple___healthy |
| 4 | Blueberry___healthy |
| 5 | Cherry_(including_sour)___Powdery_mildew |
| 6 | Cherry_(including_sour)___healthy |
| 7 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |
| 8 | Corn_(maize)___Common_rust_ |
| 9 | Corn_(maize)___Northern_Leaf_Blight |
| 10 | Corn_(maize)___healthy |
| 11 | Grape___Black_rot |
| 12 | Grape___Esca_(Black_Measles) |
| 13 | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) |
| 14 | Grape___healthy |
| 15 | Orange___Haunglongbing_(Citrus_greening) |
| 16 | Peach___Bacterial_spot |
| 17 | Peach___healthy |
| 18 | Pepper,_bell___Bacterial_spot |
| 19 | Pepper,_bell___healthy |
| 20 | Potato___Early_blight |
| 21 | Potato___Late_blight |
| 22 | Potato___healthy |
| 23 | Raspberry___healthy |
| 24 | Soybean___healthy |
| 25 | Squash___Powdery_mildew |
| 26 | Strawberry___Leaf_scorch |
| 27 | Strawberry___healthy |
| 28 | Tomato___Bacterial_spot |
| 29 | Tomato___Early_blight |
| 30 | Tomato___Late_blight |
| 31 | Tomato___Leaf_Mold |
| 32 | Tomato___Septoria_leaf_spot |
| 33 | Tomato___Spider_mites Two-spotted_spider_mite |
| 34 | Tomato___Target_Spot |
| 35 | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 36 | Tomato___Tomato_mosaic_virus |
| 37 | Tomato___healthy |

---

## 📊 Expected Performance

| Metric | Expected |
|--------|----------|
| Validation Accuracy | ~93-97% |
| Training Time (CPU) | ~2-4 hours |
| Training Time (GPU) | ~15-30 minutes |
| Model PKL Size | ~50-100 MB |
