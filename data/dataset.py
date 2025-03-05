import numpy as np
import os
import torch
import PIL.Image
from PIL import Image
import io
import urllib

from tqdm import tqdm
from torch.utils.data import Dataset

class ConceptualCaptionsDataset(Dataset):
    def __init__(
        self,
        dataset,
        original_indices,
        cache_dir="dataset",
        transform=None,
        first_run=False
    ):
        """
        Args:
            dataset: Hugging Face dataset with image URLs and captions.
            cache_dir: Directory to store downloaded images.
            transform: Optional image transformations.
        """
        self.dataset = dataset
        self.original_indices = original_indices
        self.cache_dir = cache_dir
        self.transform = transform

        if first_run:
            self._prevalidate_images()

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"found {len(self.original_indices)} images in the dataset")

    def fetch_image(self, image_url, image_id):
        """Downloads image if not cached, otherwise loads from disk."""
        image_path = os.path.join(self.cache_dir, f"{image_id}.jpg")

        # If image exists, load from disk
        return PIL.Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None

        # raise ValueError(f"{image_path} does not exist please set prevalidation=True"
        #                      "to delete corrupt images")
        #
        # # Otherwise, download and save it
        # try:
        #     request = urllib.request.Request(image_url, headers={"user-agent": "datasets"})
        #     with urllib.request.urlopen(request, timeout=5) as req:
        #         image = PIL.Image.open(io.BytesIO(req.read())).convert("RGB")
        #         image.save(image_path, "JPEG")  # Save to cache
        #     return image
        # except Exception:
        #     return None  # Return None for failed downloads

    def get_image_path(self, idx):
        image_id = self.original_indices[idx]
        return os.path.join(self.cache_dir, f"{image_id}.jpg")

    def _prevalidate_images(self):
        for idx in tqdm(range(len(self.dataset)), desc="initial_validation"):
            original_idx = self.original_indices[idx]
            image_path = self.cache_dir / f"{original_idx}.jpg"
            try:
                with Image.open(image_path) as image:
                    image.getdata()[0]
                    if image.shape[0] ==1:
                        print(f"image wrong {image_id}")
                        os.remove(image_path)
                    elif image.shape == (1,1,3):
                        print(f"image wrong {image_id}")
                        os.remove(image_path)
                    
                # self.original_indices.remove(idx)
            except Exception as e:
                print(e)
                os.remove(image_path)

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, idx):

        # Generate a unique image ID based on index
        image_id = self.original_indices[idx]
        data = self.dataset[image_id]
        image_url = data["image_url"]
        caption = data["caption"]

        image = self.fetch_image(image_url, image_id)

        # If the image fails to load, return a blank tensor
        if image is None:
            image = PIL.Image.new("RGB", (400, 400), (255, 255, 255))  # White placeholder
            caption = "A blank white image."

        if self.transform:

            image = np.array(image)
            try:
                
                augmented = self.transform(image=image)
                image = Image.fromarray(augmented["image"])
            except Exception as e:
                with open("exceptions.txt", "a") as f:
                    f.write(f"{e} on file {image_id} with {idx}")
                    f.write("\n")
                    os.remove(os.path.join(self.cache_dir, f"{image_id}.jpg"))

                image = PIL.Image.new("RGB", (400, 400), (255, 255, 255))  # White placeholder
                caption = "A blank white image."

        return {
            "image": image,
            "text": caption,
            "image_path":f"{self.cache_dir}/{image_id}.jpg" # for janus pro only
        }

