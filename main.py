import glob
import albumentations as A
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import AutoProcessor
from datasets import load_dataset

# Set a writable cache directory
from tqdm import tqdm

from data import ConceptualCaptionsDataset
# Load dataset from Hugging Face

data_dir = Path("./dataset/")
downloaded_indices = sorted([int(p.stem) for p in data_dir.glob("*.jpg") if p.stem.isdigit()])

dataset = load_dataset("google-research-datasets/conceptual_captions", split="train")
downloaded_subset = dataset.select(downloaded_indices)

processors = [
    AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
    AutoProcessor.from_pretrained("microsoft/git-base"),
    AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
    AutoProcessor.from_pretrained("openai/clip-vit-large-patch14"),
    AutoProcessor.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers"),
    AutoProcessor.from_pretrained("google/pix2struct-large"),
    AutoProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384"),
    AutoProcessor.from_pretrained("Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"),
    AutoProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B"),
]

# Define Augmentations with Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomScale(scale_limit=0.2, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.CoarseDropout(num_holes_range=(4,8), fill="random", hole_height_range=(8,32), hole_width_range=(8,32), p=0.5)
])

# Initialize dataset with caching enabled
conceptual_dataset = ConceptualCaptionsDataset(dataset, cache_dir="dataset", transform=transform)

