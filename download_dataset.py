import os
import zipfile

# Dataset info
DATASET = "vipoooool/new-plant-diseases-dataset"
DOWNLOAD_PATH = "dataset.zip"
EXTRACT_PATH = "dataset"

def download_dataset():
    print("Downloading dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {DATASET}")

def extract_dataset():
    print("Extracting dataset...")
    with zipfile.ZipFile(DOWNLOAD_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

def main():
    if not os.path.exists(DOWNLOAD_PATH):
        download_dataset()
    else:
        print("Dataset already downloaded.")

    if not os.path.exists(EXTRACT_PATH):
        extract_dataset()
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    main()