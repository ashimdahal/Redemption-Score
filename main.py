from torch.utils.data import DataLoader
from torchvision import transforms

import albumentations as A

from transformers import AutoProcessor
from datasets import load_dataset

# Set a writable cache directory
from tqdm import tqdm

from data import ConceptualCaptionsDataset
# Load dataset from Hugging Face

dataset = load_dataset("google-research-datasets/conceptual_captions", split="train")
# Define Augmentations with Albumentations
transform = A.Compose([
    A.Resize(512,512, p=1),  # Resize to larger size
    A.RandomCrop(width=400, height=400, p=0.5),  # Crop to the final size
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomScale(scale_limit=0.2, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.CoarseDropout(num_holes_range=(4,8), fill="random", hole_height_range=(8,32), hole_width_range=(8,32), p=0.5)
])

transform = A.Compose([
    A.HorizontalFlip(p=1)
])
# Initialize dataset with caching enabled
conceptual_dataset = ConceptualCaptionsDataset(dataset, cache_dir="dataset", transform=transform)

# Use PyTorch DataLoader
dataloader = DataLoader(conceptual_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# Example batch retrieval
for images, captions in tqdm(dataloader):
    continue
