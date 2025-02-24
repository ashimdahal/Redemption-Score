import os

import urllib
import io
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

from PIL import Image

# Configuration
DATASET_NAME = "google-research-datasets/conceptual_captions"
SAVE_DIR = "dataset"
NUM_WORKERS = 8  # Increase based on your bandwidth
RETRIES = 1
USER_AGENT = get_datasets_user_agent()

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
            request = urllib.request.Request(
                url, 
                data=None,
                headers={"user-agent":USER_AGENT}
               )
            with urllib.request.urlopen(request, timeout=20) as req: 
                image = Image.open(io.BytesIO(req.read()))
                image.save(filename)
            return
        except Exception as e:
            pass

# Create list of (index, url) pairs
tasks = [(idx, item["image_url"]) for idx, item in enumerate(dataset)]

# Download in parallel
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    list(tqdm(executor.map(download_image, tasks), total=len(tasks)))

