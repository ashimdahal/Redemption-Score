import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset

# Configuration
DATASET_NAME = "google-research-datasets/conceptual_captions"
SAVE_DIR = "dataset"
NUM_WORKERS = 40  # Increase based on your bandwidth
RETRIES = 3

# Load dataset
dataset = load_dataset(DATASET_NAME, split="train")

# Create image directory
os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(args):
    idx, url = args
    filename = os.path.join(SAVE_DIR, f"{idx}.jpg")
    
    # Skip if already downloaded
    if os.path.exists(filename):
        return
    
    # Download with retries
    for _ in range(RETRIES):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return
        except Exception as e:
            pass

# Create list of (index, url) pairs
tasks = [(idx, item["image_url"]) for idx, item in enumerate(dataset)]

# Download in parallel
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    list(tqdm(executor.map(download_image, tasks), total=len(tasks)))
