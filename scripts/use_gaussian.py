import os, time
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import json

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
device = DEVICE
IMG_BS       = 256     
TXT_BS       = 256   
EST_BS       = 2048  
NUM_IMG_W    = 2     
PIN_MEMORY   = True   
SAVE_TENSORS = True
TEST_BS = 512

def safe_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        return None


class CaptionDataset(Dataset):
    def __init__(self, captions):  self.caps = captions
    def __len__(self):             return len(self.caps)
    def __getitem__(self, i):      return self.caps[i]

class UniqueImageDataset(Dataset):
    """Iterates over *unique* image paths; returns (index, PIL_image)."""
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):             return len(self.paths)
    def __getitem__(self, idx):    return idx, safe_image(self.paths[idx])


from transformers import CLIPTokenizerFast

clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14-336")

def collate_txt(batch):
    return clip_tokenizer(
        batch,
        padding='max_length',     # pad to fixed size
        max_length=77,            # truncate to 77 tokens (CLIP’s max)
        truncation=True,          # this is the critical fix!
        return_tensors="pt"
    )

def load_gaussian(path: str = "data/distributions/mid_gaussian.pt") -> dict:
    """Returns the Gaussian dict on CPU."""
    return torch.load(path, map_location="cpu")

def get_clip():
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE)
    model.eval()
    return proc, model

def collate_img(batch):
    """Batch = list[(idx, PIL)].  Returns (idx_tensor, list[PIL])."""
    idxs, imgs = zip(*batch)
    return torch.tensor(idxs, dtype=torch.long), list(imgs)

def pmi_scores_batch(xb: torch.Tensor, yb: torch.Tensor, dists: dict, tau=1.0) -> torch.Tensor:
    """
    xb, yb: (B, D) torch.float64
    dists: dict from estimator.finalize()
    returns: (B,) tensor of PMI scores
    """

    dev = dists["mu_x"].device            # e.g. cuda:0
    xb  = xb.to(dev, dtype=torch.float64)
    yb  = yb.to(dev, dtype=torch.float64)

    
    μ_x, μ_y, μ_z = dists["mu_x"], dists["mu_y"], dists["mu_z"]
    invΣ_x, invΣ_y, invΣ_z = dists["invΣ_x"], dists["invΣ_y"], dists["invΣ_z"]
    ld_x, ld_y, ld_z = dists["logdet_x"], dists["logdet_y"], dists["logdet_z"]

    dx = xb - μ_x
    dy = yb - μ_y
    dz = torch.cat([dx, dy], dim=1)  # (B, 2D)

    # squared Mahalanobis distances:
    d2_x = (dx @ invΣ_x * dx).sum(dim=1)
    d2_y = (dy @ invΣ_y * dy).sum(dim=1)
    d2_z = (dz @ invΣ_z * dz).sum(dim=1)

    mi_const = (ld_x + ld_y - ld_z) / 2
    return mi_const + 0.5/tau * (d2_x + d2_y - d2_z)



def main(samples_json_path):
    with open(samples_json_path, 'r') as f:
        all_samples = json.load(f)

    dists = load_gaussian("mid_gaussian_coco")

    for model_name, samples in tqdm(all_samples.items(), desc="Models"):
        # preds = [s["generated_caption"] for s in samples][:save_samples]
        # path = ["root_folder" +"/"+ "predictions" + "/" +model_name +"/"+ s['image_path'].split("/"[1]) for s in samples]
        test_images = [s["image_path"] for s in samples]
        test_captions = [s["generated_caption"]  for s in samples]

        img_test_ds = UniqueImageDataset(test_images)
        img_test_dl = DataLoader(img_test_ds,
                                 batch_size   = TEST_BS,
                                 shuffle      = False,
                                 num_workers  = NUM_IMG_W,
                                 pin_memory   = PIN_MEMORY,
                                 collate_fn   = collate_img)


        txt_test_ds = CaptionDataset(test_captions)
        txt_test_dl = DataLoader(txt_test_ds,
                                 batch_size   = TEST_BS,
                                 shuffle      = False,
                                 num_workers  = 1,
                                 collate_fn   = collate_txt)

        pair_test_dl = zip(img_test_dl, txt_test_dl)

        proc, model = get_clip()

        pmi_scores = []
        with torch.inference_mode():
            for (idxs, imgs), caps in tqdm(pair_test_dl, desc=f"{model_name}"):
                # inputs = proc(text=caps, images=imgs,
                #               return_tensors="pt", padding=True)
                inputs = proc(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
                inputs["input_ids"] = caps["input_ids"].to(DEVICE)
                inputs["attention_mask"] = caps["attention_mask"].to(DEVICE)

                raw_img_f = model.get_image_features(inputs["pixel_values"].to(DEVICE)).cpu().double()
                raw_txt_f = model.get_text_features(
                            inputs["input_ids"].to(DEVICE),
                            attention_mask=inputs["attention_mask"].to(DEVICE)
                        ).cpu().double()

                img_f = torch.nn.functional.normalize(raw_img_f, p=2, dim=-1)
                txt_f = torch.nn.functional.normalize(raw_txt_f, p=2, dim=-1)
                tau = 1.0    # e.g. 1.0 for Flickr8K, 0.1 for FOIL
                scores = pmi_scores_batch(img_f, txt_f, dists)
                pmi_scores.extend(pmi_scores_batch(img_f, txt_f, dists).tolist())


        print(len(pmi_scores), "PMI scores computed")
        with open(f"pmi_{model_name}", "w", encoding="utf‑8") as f:
            json.dump(pmi_scores, f, indent=2)   # indent=2 → nicely formatted

if __name__ == "__main__":
    main("output_coco.json")
