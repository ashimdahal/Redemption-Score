from pathlib import Path
from datasets import load_dataset

import albumentations as A
from data import ConceptualCaptionsDataset

data_dir = Path("./dataset/")

dataset = load_dataset("google-research-datasets/conceptual_captions", split="train")
print("checking downloaded files")
downloaded_indices = sorted([int(p.stem) for p in data_dir.glob("*.jpg") if p.stem.isdigit()])
print(f"found {len(downloaded_indices)} predownloaded images, selecting them as the subset")

downloaded_subset = dataset.select(downloaded_indices)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
])

print("cleaning dataset")
dataset = ConceptualCaptionsDataset(
    downloaded_subset,
    downloaded_indices,
    cache_dir=data_dir,
    transform=transform,
    first_run=True
)

print("finished cleaning dataset")
