"""
PlantVillage Disease Detection - Flask Web App
================================================
Deploys the trained CNN model as a web API + UI.

Usage:
    py -3.13 app.py
    
Then open: http://localhost:5000
"""

import os
import sys
import json
import io
import base64
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PKL_PATH = os.path.join(OUTPUT_DIR, "plant_disease_model.pkl")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ============================================================
# Load Model
# ============================================================
def load_model():
    """Load trained model from .pkl file."""
    print("[INFO] Loading model...")
    model_package = joblib.load(PKL_PATH)
    
    num_classes = model_package['num_classes']
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
    
    model.load_state_dict(model_package['model_state_dict'])
    model.eval()
    
    print(f"[INFO] Model loaded: {model_package['model_name']}")
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Accuracy: {model_package['accuracy']*100:.2f}%")
    
    return model, model_package

# Load on startup
model, model_package = load_model()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Disease info database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "description": "A fungal disease caused by Venturia inaequalis that creates dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides like captan or myclobutanil. Remove fallen leaves. Plant resistant varieties.",
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "description": "Caused by the fungus Botryosphaeria obtusa. Creates brown lesions with dark borders on leaves.",
        "treatment": "Prune dead wood, remove mummified fruits. Apply fungicides during growing season.",
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "A fungal disease requiring both apple and cedar/juniper trees to complete its lifecycle.",
        "treatment": "Remove nearby cedar/juniper trees. Apply protective fungicides in spring.",
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "description": "A fungal disease that creates white powdery spots on leaves, reducing photosynthesis.",
        "treatment": "Apply sulfur-based fungicides. Improve air circulation. Remove infected parts.",
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": "The leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
}

def predict_image(image_bytes):
    """Run prediction on image bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    
    top5_probs, top5_indices = probs.topk(min(5, len(probs)))
    idx_to_class = model_package.get('idx_to_class', {})
    
    results = []
    for prob, idx in zip(top5_probs, top5_indices):
        class_name = idx_to_class.get(str(idx.item()), idx_to_class.get(idx.item(), f"Class_{idx.item()}"))
        info = DISEASE_INFO.get(class_name, {})
        results.append({
            'class_name': class_name,
            'confidence': round(prob.item() * 100, 2),
            'plant': info.get('plant', class_name.split('___')[0].replace('_', ' ') if '___' in class_name else 'Unknown'),
            'disease': info.get('disease', class_name.split('___')[1].replace('_', ' ') if '___' in class_name else 'Unknown'),
            'description': info.get('description', ''),
            'treatment': info.get('treatment', ''),
            'severity': info.get('severity', 'Unknown'),
        })
    
    return results

# ============================================================
# HTML Template
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlantVillage Disease Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0f1a;
            --bg-secondary: #111827;
            --bg-card: #1a2332;
            --bg-card-hover: #1f2b3d;
            --accent-green: #10b981;
            --accent-emerald: #34d399;
            --accent-teal: #14b8a6;
            --accent-red: #ef4444;
            --accent-orange: #f59e0b;
            --accent-blue: #3b82f6;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #1e293b;
            --glass-bg: rgba(17, 24, 39, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated background */
        .bg-pattern {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 50%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(20, 184, 166, 0.05) 0%, transparent 50%);
            z-index: 0;
            animation: bgShift 20s ease-in-out infinite alternate;
        }
        
        @keyframes bgShift {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }
        
        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 24px 20px;
            position: relative;
            z-index: 1;
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 0 20px;
        }
        
        .header .logo {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-teal));
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            margin: 0 auto 20px;
            box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }
        
        .header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-emerald), var(--accent-teal), var(--accent-blue));
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        
        .header p {
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 400;
        }
        
        .model-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 20px;
            font-size: 0.8rem;
            color: var(--accent-emerald);
            margin-top: 12px;
        }
        
        .model-badge .dot {
            width: 7px; height: 7px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        /* Upload Area */
        .upload-section {
            background: var(--bg-card);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
        }
        
        .drop-zone {
            border: 2px dashed var(--border-color);
            border-radius: 16px;
            padding: 50px 30px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .drop-zone:hover, .drop-zone.dragover {
            border-color: var(--accent-green);
            background: rgba(16, 185, 129, 0.04);
        }
        
        .drop-zone .icon {
            font-size: 52px;
            margin-bottom: 16px;
            display: block;
        }
        
        .drop-zone h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-primary);
        }
        
        .drop-zone p {
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        .drop-zone .browse-btn {
            display: inline-block;
            margin-top: 16px;
            padding: 10px 28px;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-teal));
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        
        .drop-zone .browse-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }
        
        #fileInput { display: none; }
        
        /* Preview */
        .preview-container {
            display: none;
            margin-top: 24px;
        }
        
        .preview-container.active { display: block; }
        
        .preview-img-wrap {
            position: relative;
            display: inline-block;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }
        
        .preview-img-wrap img {
            max-width: 320px;
            max-height: 280px;
            display: block;
            border-radius: 16px;
        }
        
        .predict-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 20px;
            padding: 14px 36px;
            background: linear-gradient(135deg, var(--accent-green), #059669);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
            font-family: 'Inter', sans-serif;
        }
        
        .predict-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(16, 185, 129, 0.5);
        }
        
        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        /* Loading spinner */
        .spinner {
            display: none;
            width: 20px; height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }
        
        .spinner.active { display: inline-block; }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Results */
        .results-section {
            display: none;
            animation: slideUp 0.5s ease;
        }
        
        .results-section.active { display: block; }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-card {
            background: var(--bg-card);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 20px;
            backdrop-filter: blur(20px);
        }
        
        .result-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .result-icon {
            width: 56px; height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            flex-shrink: 0;
        }
        
        .result-icon.healthy {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(52, 211, 153, 0.1));
        }
        
        .result-icon.diseased {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(245, 158, 11, 0.1));
        }
        
        .result-header h2 {
            font-size: 1.4rem;
            font-weight: 700;
        }
        
        .result-header .confidence {
            font-size: 0.9rem;
            color: var(--accent-emerald);
            font-weight: 600;
        }
        
        .severity-badge {
            display: inline-flex;
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .severity-badge.none {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
        }
        
        .severity-badge.moderate {
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-orange);
        }
        
        .severity-badge.high {
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-red);
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 20px;
        }
        
        .info-box {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 16px;
        }
        
        .info-box h4 {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .info-box p {
            font-size: 0.88rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }
        
        /* Confidence bars */
        .confidence-list {
            margin-top: 20px;
        }
        
        .confidence-item {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        
        .confidence-item .label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            width: 240px;
            flex-shrink: 0;
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .confidence-bar-bg {
            flex: 1;
            height: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .confidence-bar {
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-teal));
        }
        
        .confidence-item .pct {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-primary);
            width: 50px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 30px 0;
            color: var(--text-muted);
            font-size: 0.8rem;
        }
        
        /* API Info */
        .api-section {
            background: var(--bg-card);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            margin-top: 20px;
        }
        
        .api-section h3 {
            font-size: 1rem;
            margin-bottom: 12px;
            color: var(--text-primary);
        }
        
        .code-block {
            background: var(--bg-primary);
            border-radius: 10px;
            padding: 16px;
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            color: var(--accent-emerald);
            overflow-x: auto;
            line-height: 1.6;
        }
        
        @media (max-width: 640px) {
            .info-grid { grid-template-columns: 1fr; }
            .confidence-item .label { width: 140px; font-size: 0.7rem; }
            .header h1 { font-size: 1.6rem; }
            .upload-section { padding: 24px; }
        }
    </style>
</head>
<body>
    <div class="bg-pattern"></div>
    
    <div class="container">
        <header class="header">
            <div class="logo">🌿</div>
            <h1>Plant Disease Detector</h1>
            <p>AI-powered plant disease classification using deep learning</p>
            <div class="model-badge">
                <span class="dot"></span>
                MobileNetV2 · {{ accuracy }}% Accuracy · {{ num_classes }} Classes
            </div>
        </header>
        
        <section class="upload-section">
            <div class="drop-zone" id="dropZone">
                <span class="icon">📸</span>
                <h3>Upload a leaf image</h3>
                <p>Drag & drop or click to browse · JPG, PNG supported</p>
                <button class="browse-btn" onclick="document.getElementById('fileInput').click()">
                    Browse Files
                </button>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="preview-container" id="previewContainer">
                <div class="preview-img-wrap">
                    <img id="previewImg" src="" alt="Preview">
                </div>
                <br>
                <button class="predict-btn" id="predictBtn" onclick="predict()">
                    <span id="btnText">🔍 Analyze Disease</span>
                    <div class="spinner" id="spinner"></div>
                </button>
            </div>
        </section>
        
        <section class="results-section" id="resultsSection">
            <div class="result-card" id="mainResult"></div>
            <div class="result-card">
                <h3 style="font-size:1rem; margin-bottom:16px; color:var(--text-secondary);">All Predictions</h3>
                <div class="confidence-list" id="confidenceList"></div>
            </div>
        </section>
        
        <section class="api-section">
            <h3>🔌 API Endpoint</h3>
            <div class="code-block">
POST /predict<br>
Content-Type: multipart/form-data<br><br>
curl -X POST -F "file=@leaf.jpg" http://localhost:5000/predict
            </div>
        </section>
        
        <footer class="footer">
            PlantVillage Disease Detector · MobileNetV2 Transfer Learning · PyTorch
        </footer>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImg = document.getElementById('previewImg');
        const resultsSection = document.getElementById('resultsSection');
        let selectedFile = null;
        
        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFile(files[0]);
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file');
                return;
            }
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewContainer.classList.add('active');
                resultsSection.classList.remove('active');
            };
            reader.readAsDataURL(file);
        }
        
        async function predict() {
            if (!selectedFile) return;
            
            const btn = document.getElementById('predictBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');
            
            btn.disabled = true;
            btnText.textContent = 'Analyzing...';
            spinner.classList.add('active');
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data.predictions);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (err) {
                alert('Connection error: ' + err.message);
            } finally {
                btn.disabled = false;
                btnText.textContent = '🔍 Analyze Disease';
                spinner.classList.remove('active');
            }
        }
        
        function displayResults(predictions) {
            const top = predictions[0];
            const isHealthy = top.disease.toLowerCase().includes('healthy');
            
            const mainResult = document.getElementById('mainResult');
            const severityClass = top.severity === 'None' ? 'none' : 
                                  top.severity === 'High' ? 'high' : 'moderate';
            
            mainResult.innerHTML = `
                <div class="result-header">
                    <div class="result-icon ${isHealthy ? 'healthy' : 'diseased'}">
                        ${isHealthy ? '✅' : '⚠️'}
                    </div>
                    <div>
                        <h2>${isHealthy ? 'Healthy Plant' : top.disease}</h2>
                        <span class="confidence">${top.confidence}% confidence · ${top.plant}</span>
                    </div>
                    <span class="severity-badge ${severityClass}" style="margin-left:auto;">
                        ${top.severity === 'None' ? '● Healthy' : '● ' + top.severity}
                    </span>
                </div>
                <div class="info-grid">
                    <div class="info-box">
                        <h4>📋 Description</h4>
                        <p>${top.description || 'No description available.'}</p>
                    </div>
                    <div class="info-box">
                        <h4>💊 Treatment</h4>
                        <p>${top.treatment || 'No treatment info available.'}</p>
                    </div>
                </div>
            `;
            
            const confidenceList = document.getElementById('confidenceList');
            confidenceList.innerHTML = predictions.map(p => `
                <div class="confidence-item">
                    <span class="label">${p.disease} (${p.plant})</span>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar" style="width: ${p.confidence}%"></div>
                    </div>
                    <span class="pct">${p.confidence}%</span>
                </div>
            `).join('');
            
            resultsSection.classList.add('active');
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    </script>
</body>
</html>
"""

# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, 
        accuracy=round(model_package['accuracy'] * 100, 2),
        num_classes=model_package['num_classes']
    )

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        image_bytes = file.read()
        predictions = predict_image(image_bytes)
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': model_package['model_name'],
        'accuracy': model_package['accuracy'],
        'classes': model_package['num_classes']
    })

@app.route('/classes')
def classes():
    return jsonify({
        'classes': model_package['class_names'],
        'class_to_idx': model_package['class_to_idx']
    })


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  🌿 Plant Disease Detector - Web App")
    print(f"{'='*60}")
    print(f"  Model: {model_package['model_name']}")
    print(f"  Accuracy: {model_package['accuracy']*100:.2f}%")
    print(f"  Classes: {model_package['num_classes']}")
    print(f"\n  🌐 Open: http://localhost:5000")
    print(f"  📡 API:  POST http://localhost:5000/predict")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
