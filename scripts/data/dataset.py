import numpy as np
import os
import torch
import PIL.Image
from PIL import Image
import io
import urllib
from pathlib import Path

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
                    image = np.array(image)
                    if image.shape[0] ==1:
                        print(f"image wrong {original_idx}")
                        os.remove(image_path)
                    elif image.shape == (1,1,3):
                        print(f"image wrong {original_idx}")
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


class CocoDataset(Dataset):
    def __init__(
        self,
        images_base_dir,         # Path to the directory containing COCO images (e.g., ./coco_valid_dataset/)
        annotations_json_file, # Path to captions_val2017.json
        transform=None,
        # Optional: if you want to limit the number of items loaded by the dataset itself.
        # Otherwise, it loads all valid items and evaluation script can take a subset.
        max_items_to_load=None
    ):
        """
        Args:
            images_base_dir: Directory where COCO image files (e.g., 000000xxxxxx.jpg) are stored.
            annotations_json_file: Path to the COCO annotations JSON file (e.g., captions_val2017.json).
            transform: Optional image transformations.
            max_items_to_load: Optionally cap the number of items loaded into the dataset.
        """
        self.images_base_dir = Path(images_base_dir)
        self.annotations_json_file = Path(annotations_json_file)
        self.transform = transform
        self.internal_data_list = [] # Will store {"image_id": ..., "file_name": ..., "caption": ...}

        if not self.images_base_dir.exists() or not self.images_base_dir.is_dir():
            raise FileNotFoundError(f"COCO images directory not found: {self.images_base_dir}")
        if not self.annotations_json_file.exists():
            raise FileNotFoundError(f"COCO annotations JSON file not found: {self.annotations_json_file}")

        print(f"Loading COCO annotations from: {self.annotations_json_file}")
        with open(self.annotations_json_file, 'r') as f:
            coco_raw_data = json.load(f)

        # Create a map of image_id to its metadata (like file_name)
        image_id_to_meta = {img_info['id']: img_info for img_info in coco_raw_data.get('images', [])}

        # Create a map of image_id to a list of its captions
        image_id_to_captions_texts = {}
        for ann_info in coco_raw_data.get('annotations', []):
            img_id = ann_info['image_id']
            if img_id not in image_id_to_captions_texts:
                image_id_to_captions_texts[img_id] = []
            image_id_to_captions_texts[img_id].append(ann_info['caption'])
        
        print(f"Found metadata for {len(image_id_to_meta)} images and captions for {len(image_id_to_captions_texts)} images in JSON.")

        # Populate internal_data_list, ensuring image files exist and have captions
        # Iterate through image metadata to ensure we are processing defined images
        items_added = 0
        for img_id, img_meta in tqdm(image_id_to_meta.items(), desc="Verifying images and captions"):
            if max_items_to_load is not None and items_added >= max_items_to_load:
                break

            file_name = img_meta.get('file_name')
            if not file_name:
                # print(f"Warning: Image ID {img_id} has no file_name in annotations. Skipping.")
                continue

            image_file_path = self.images_base_dir / file_name
            if not image_file_path.exists():
                # This can be very verbose if many COCO images are not downloaded for a split.
                # print(f"Warning: Image file {image_file_path} for ID {img_id} not found. Skipping.")
                continue
            
            captions_for_image = image_id_to_captions_texts.get(img_id)
            if not captions_for_image or not captions_for_image[0]:
                # print(f"Warning: No captions found for Image ID {img_id} ({file_name}). Skipping.")
                continue
            
            self.internal_data_list.append({
                "image_id_original": img_id,
                "file_name": file_name, # Original COCO filename
                "caption": captions_for_image[0] # Take the first caption
            })
            items_added += 1
        
        if not self.internal_data_list:
            raise ValueError(f"No valid image-caption pairs found. Check image directory '{self.images_base_dir}' and annotations '{self.annotations_json_file}'.")
        
        if max_items_to_load is not None and items_added < max_items_to_load:
            print(f"Warning: Loaded only {items_added} items out of requested {max_items_to_load} due to missing files or captions.")

        print(f"CocoDataset initialized with {len(self.internal_data_list)} valid image-caption pairs.")


    def __len__(self):
        return len(self.internal_data_list)

    def get_image_path(self, idx):
        """Gets the full path to the image file for the given dataset index."""
        if not (0 <= idx < len(self.internal_data_list)):
            raise IndexError(f"Index {idx} is out of bounds for internal data list of length {len(self.internal_data_list)}")
        
        item_info = self.internal_data_list[idx]
        return str(self.images_base_dir / item_info["file_name"])

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.internal_data_list)):
            raise IndexError(f"Index {idx} is out of bounds for internal data list of length {len(self.internal_data_list)}")

        item_info = self.internal_data_list[idx]
        
        file_name = item_info["file_name"]
        caption = item_info["caption"]
        full_image_path = self.images_base_dir / file_name
        
        image = None
        try:
            image = PIL.Image.open(full_image_path).convert("RGB")
        except FileNotFoundError:
            print(f"ERROR: Image file not found at {full_image_path} for index {idx}. This should have been caught in __init__.")
            # Fallback, though __init__ should prevent this
            image = PIL.Image.new("RGB", (400, 400), (128, 128, 128)) # Grey placeholder
            caption = "Error: Image file missing post-init."
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            image = PIL.Image.new("RGB", (400, 400), (128, 128, 128))
            caption = f"Error loading image: {e}"


        if self.transform and image:
            image_np = np.array(image)
            try:
                augmented = self.transform(image=image_np)
                image = Image.fromarray(augmented["image"])
            except Exception as e:
                # Log exception
                error_log_file = "transform_exceptions_coco.txt"
                with open(error_log_file, "a") as f:
                    f.write(f"Error applying transform to {full_image_path} (index: {idx}): {e}\n")
                
                # No os.remove here as these are original files
                print(f"Warning: Transform error for {full_image_path}. Using placeholder image.")
                image = PIL.Image.new("RGB", (400, 400), (0, 0, 0))  # Black placeholder on transform error
                caption = "A black image (transform error)."

        return {
            "image": image,
            "text": caption,
            "image_path": str(full_image_path) # Return the full, original path
            # Optional: include original_image_id if needed elsewhere
            # "original_image_id": item_info.get("image_id_original")
        }

