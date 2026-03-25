"""
PlantVillage Dataset Downloader
================================
Downloads the PlantVillage dataset from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Place your kaggle.json API key in:
   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
   - Linux: ~/.kaggle/kaggle.json
   
   Get your API key from: https://www.kaggle.com/settings -> "Create New Token"
"""

import os
import sys
import zipfile
import shutil

def download_dataset():
    """Download and extract the PlantVillage dataset."""
    
    # Set dataset path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        print(f"[INFO] Dataset directory already exists at: {dataset_dir}")
        print("[INFO] Skipping download. Delete the 'dataset' folder to re-download.")
        return dataset_dir
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] Kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)
    
    print("[INFO] Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    print("[INFO] Downloading PlantVillage dataset (~4.3 GB)...")
    print("[INFO] This may take a while depending on your internet speed...")
    
    api.dataset_download_files(
        dataset="abdallahalidev/plantvillage-dataset",
        path=dataset_dir,
        unzip=True
    )
    
    print(f"[SUCCESS] Dataset downloaded and extracted to: {dataset_dir}")
    
    # Show dataset structure
    print("\n[INFO] Dataset structure:")
    for root, dirs, files in os.walk(dataset_dir):
        level = root.replace(dataset_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level >= 2:  # Don't go too deep
            continue
        subindent = ' ' * 2 * (level + 1)
        for d in sorted(dirs):
            print(f'{subindent}{d}/')
    
    return dataset_dir


if __name__ == "__main__":
    download_dataset()
