import os
import json
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CaptionImageDataset(Dataset):
    """Dataset returning (caption, output_path) pairs."""
    def __init__(self, captions, output_paths):
        assert len(captions) == len(output_paths)
        self.captions = captions
        self.paths = output_paths

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx], self.paths[idx]

def generate_images_dataloader(pipe, captions, output_paths,
                               num_inference_steps=50, batch_size=16):
    dataset = CaptionImageDataset(captions, output_paths)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for batch_captions, batch_paths in tqdm(loader, desc="Generating images"):
        images = pipe(prompt=list(batch_captions),
                      num_inference_steps=num_inference_steps).images
        for img, path in zip(images, batch_paths):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)
        torch.cuda.empty_cache()

def setup_models():
    """Set up SD3-Medium and CLIP (if you need it later)."""
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()

    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return pipe, clip_model, clip_processor

def process_simple_json(json_path, output_dir,
                        batch_size=16, num_inference_steps=50):
    # load your new-format JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    pipe, clip_model, clip_processor = setup_models()

    for key, list_name in (("reference", "references"), ("pred", "predictions")):
        items = data.get(key, [])
        if not items:
            continue

        captions    = [item["caption"] for item in items]
        folder_path = os.path.join(output_dir, list_name)
        os.makedirs(folder_path, exist_ok=True)

        output_paths = []
        for item in items:
            # strip off everything after '#' and '.jpg'
            file_part = item["caption_id"].split("#")[0]            # e.g. "1056338697_4f7d7ce270.jpg"
            num       = os.path.splitext(file_part)[0]             # e.g. "1056338697_4f7d7ce270"
            filename  = f"{num}_{list_name}.png"                   # e.g. "1056338697_4f7d7ce270_references.jpg"
            output_paths.append(os.path.join(folder_path, filename))

        generate_images_dataloader(
            pipe,
            captions,
            output_paths,
            num_inference_steps=num_inference_steps,
            batch_size=batch_size
        )

if __name__ == "__main__":
    process_simple_json(
        json_path="flickr.json",
        output_dir="model_comparison_batch_sdx1",
        batch_size=16,
        num_inference_steps=40
    )

