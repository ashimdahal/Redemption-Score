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
        hf_dataset_slice, # This should be the Hugging Face dataset object, sliced to the desired number of items
        original_indices, # List of integer indices (e.g., [0, 1, ..., 2999]) corresponding to saved image names
        cache_dir="coco_valid_dataset_3k", # Directory where images like "0.jpg", "1.jpg" are stored
        transform=None,
        # first_run=False # Optional: if you want to include prevalidation
    ):
        """
        Args:
            hf_dataset_slice: Hugging Face dataset object (e.g., COCO 2017 val, sliced).
            original_indices: List of integer indices that correspond to the filenames (e.g., 0 for 0.jpg)
                              and are used to index into hf_dataset_slice.
            cache_dir: Directory where pre-downloaded images are stored.
            transform: Optional image transformations.
        """
        self.dataset = hf_dataset_slice
        self.original_indices = original_indices # These are the indices from the slice, used as filenames
        self.cache_dir = Path(cache_dir) # Ensure cache_dir is a Path object
        self.transform = transform

        # Optional: Prevalidation logic (adapted from your ConceptualCaptionsDataset)
        # if first_run:
        #     self._prevalidate_images()

        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"CocoDataset initialized. Found {len(self.original_indices)} image indices to use from '{self.cache_dir}'.")

    def fetch_image_from_cache(self, image_filename_stem):
        """Loads a pre-downloaded image from the cache."""
        image_path = os.path.join(self.cache_dir, f"{image_filename_stem}.jpg")
        return PIL.Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None

    def get_image_path(self, idx):
        """Gets the full path to the image file for the given dataset index."""
        # 'idx' is the index for the DataLoader (0 to len(self)-1)
        # self.original_indices[idx] gives the filename stem (e.g., 0, 1, ...)
        image_filename_stem = self.original_indices[idx]
        return os.path.join(self.cache_dir, f"{image_filename_stem}.jpg")

    # Optional: Adapted prevalidation logic
    # def _prevalidate_images(self):
    #     print(f"Starting prevalidation of images in {self.cache_dir}...")
    #     valid_indices_after_check = []
    #     for original_idx_val in tqdm(self.original_indices, desc="Initial validation of COCO images"):
    #         image_path = self.cache_dir / f"{original_idx_val}.jpg"
    #         if not image_path.exists():
    #             print(f"Image {image_path} not found during prevalidation. Skipping index {original_idx_val}.")
    #             continue
    #         try:
    #             with Image.open(image_path) as img:
    #                 img.verify() # Basic check
    #                 # More thorough check by loading data
    #                 with Image.open(image_path) as img_load:
    #                     img_load.getdata()[0]
    #                     img_np = np.array(img_load)
    #                     if img_np.shape[0] == 1 or img_np.shape == (1,1,3) or min(img_np.shape[:2]) < 10: # Example checks
    #                         print(f"Image {image_path} (original_idx: {original_idx_val}) seems problematic (shape: {img_np.shape}). Removing.")
    #                         os.remove(image_path)
    #                         continue # Skip adding this index
    #             valid_indices_after_check.append(original_idx_val)
    #         except Exception as e:
    #             print(f"Error validating image {image_path} (original_idx: {original_idx_val}): {e}. Removing.")
    #             try:
    #                 os.remove(image_path)
    #             except OSError as oe:
    #                 print(f"Could not remove problematic image {image_path}: {oe}")
    #     # self.original_indices = valid_indices_after_check # This would change the dataset length
    #     # print(f"Prevalidation complete. {len(self.original_indices)} images remain.")


    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, idx):
        # idx is the index requested by the DataLoader, from 0 to len(self)-1.
        # self.original_indices[idx] maps this to the specific file stem (e.g., 0, 1, ...)
        # and also the index into the hf_dataset_slice.
        item_hf_idx = self.original_indices[idx] # This is the index for hf_dataset_slice and the image filename stem.

        # Fetch corresponding item from the Hugging Face dataset slice
        try:
            data_item_from_hf = self.dataset[item_hf_idx]
        except IndexError:
            # This should not happen if original_indices are correctly aligned with hf_dataset_slice
            print(f"Error: Index {item_hf_idx} out of bounds for the provided HF dataset slice.")
            image = PIL.Image.new("RGB", (400, 400), (128, 128, 128)) # Grey placeholder
            caption = "Error: Could not retrieve data from HF dataset."
            img_path = f"{self.cache_dir}/error_idx_{item_hf_idx}.jpg"
            return {
                "image": image, "text": caption, "image_path": img_path
            }

        # Get the first caption. COCO structure from "ydshieh/coco_dataset_script"
        # has captions as a list of dictionaries.
        try:
            caption = data_item_from_hf['captions'][0]['caption']
        except (IndexError, KeyError, TypeError) as e:
            print(f"Warning: Could not extract caption for item_hf_idx {item_hf_idx} (COCO file_name: {data_item_from_hf.get('file_name', 'N/A')}). Error: {e}. Using placeholder.")
            caption = "Caption not available."

        # Load the pre-downloaded image from cache
        image = self.fetch_image_from_cache(str(item_hf_idx)) # image_filename_stem is item_hf_idx

        image_path_for_results = os.path.join(self.cache_dir, f"{item_hf_idx}.jpg")

        if image is None:
            print(f"Warning: Image for item_hf_idx {item_hf_idx} (path: {image_path_for_results}) failed to load from cache. Using placeholder.")
            image = PIL.Image.new("RGB", (400, 400), (255, 255, 255))  # White placeholder
            caption = "A blank white image (image load failed)." # Overwrite caption if image failed

        if self.transform:
            image_np = np.array(image)
            try:
                augmented = self.transform(image=image_np)
                image = Image.fromarray(augmented["image"])
            except Exception as e:
                # Log exception and remove problematic file
                error_log_file = "transform_exceptions_coco.txt"
                with open(error_log_file, "a") as f:
                    f.write(f"Error applying transform to {image_path_for_results} (item_hf_idx: {item_hf_idx}): {e}\n")
                
                # Attempt to remove the problematic cached image
                try:
                    if os.path.exists(image_path_for_results):
                        os.remove(image_path_for_results)
                        print(f"Removed problematic image {image_path_for_results} due to transform error.")
                except OSError as oe:
                    print(f"Could not remove problematic image {image_path_for_results}: {oe}")

                image = PIL.Image.new("RGB", (400, 400), (0, 0, 0))  # Black placeholder on transform error
                caption = "A black image (transform error)." # Overwrite caption

        return {
            "image": image,
            "text": caption,
            "image_path": image_path_for_results
        }

