import os, time
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import json
from pathlib import Path

json_path = "/kaggle/working/Redeem/filtered.json"
IMG_BASE_DIR = "/kaggle/input/data-cc"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
device = DEVICE
IMG_BS       = 512     
TXT_BS       = 512   
EST_BS       = 2048  
NUM_IMG_W    = 2     
PIN_MEMORY   = True   
SAVE_TENSORS = True
# TEST_BS = 250

def save_gaussian(dists: dict, out_path: str = "data/distributions/mid_gaussian.pt"):
    """Move all tensors to CPU and torch.save the dict."""
    cpu_dict = {k: v.cpu() if torch.is_tensor(v) else v for k,v in dists.items()}
    save_torch(cpu_dict, out_path)

def load_gaussian(path: str = "data/distributions/mid_gaussian.pt") -> dict:
    """Returns the Gaussian dict on CPU."""
    return load_torch(path)

def save_torch(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)

class GaussianEstimator:
    def __init__(self, D, reg=5e-4, device="cpu"):
        self.D, self.reg, self.device = D, reg, device
        self.count  = 0
        self.sum_x  = torch.zeros(D, dtype=torch.float64, device=device)
        self.sum_y  = torch.zeros(D, dtype=torch.float64, device=device)
        self.sum_z  = torch.zeros(2*D, dtype=torch.float64, device=device)
        self.sum_xx = torch.zeros(D, D, dtype=torch.float64, device=device)
        self.sum_yy = torch.zeros(D, D, dtype=torch.float64, device=device)
        self.sum_zz = torch.zeros(2*D,2*D,dtype=torch.float64, device=device)
        # self.D, self.reg, self.device = D, reg, device
        # initialize sums…

    def update(self, x, y):
        x, y = x.double().to(self.device), y.double().to(self.device)
        z    = torch.cat([x, y], dim=1)
        self.count += x.size(0)
        self.sum_x  += x.sum(0);   self.sum_y  += y.sum(0);   self.sum_z  += z.sum(0)
        self.sum_xx += x.T @ x;    self.sum_yy += y.T @ y;    self.sum_zz += z.T @ z

    def finalize(self):
        N   = self.count
        μx  = self.sum_x / N;  μy = self.sum_y / N;  μz = self.sum_z / N
        Σx  = (self.sum_xx - N*μx.outer(μx))/(N-1)
        Σy  = (self.sum_yy - N*μy.outer(μy))/(N-1)
        Σz  = (self.sum_zz - N*μz.outer(μz))/(N-1)
        I   = torch.eye(self.D, device=self.device, dtype=torch.float64)
        Σx += self.reg*I
        Σy += self.reg*I
        Σz += self.reg*torch.eye(2*self.D, device=self.device, dtype=torch.float64)
        invΣx = torch.linalg.inv(Σx)
        invΣy = torch.linalg.inv(Σy)
        invΣz = torch.linalg.inv(Σz)
        ld_x  = torch.linalg.slogdet(Σx)[1];  ld_y = torch.linalg.slogdet(Σy)[1];  ld_z = torch.linalg.slogdet(Σz)[1]
        return dict(
            mu_x=μx, mu_y=μy, mu_z=μz,
            invΣ_x=invΣx, invΣ_y=invΣy, invΣ_z=invΣz,
            logdet_x=ld_x, logdet_y=ld_y, logdet_z=ld_z
        )
        
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prefix    = "/kaggle/input/data-cc/"
    paths = [prefix+e["image_path"] for e in data]
    captions = [e["caption"] for e in data]
    return data, paths, captions


# ---------- SAFE IMAGE LOAD --------------------------------------
def safe_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(e)
        print("⚠️ load error:", path, e)
        return Image.new("RGB", (224, 224))

# ---------- READ ANNOTATIONS -------------------------------------
def read_annotations(path_txt):
    img_paths, caps = [], []
    with open(path_txt, "r") as f:
        for ln in f:
            if not ln.strip():
                continue
            img_id, cap = ln.rstrip("\n").split("\t", 1)
            img_name = img_id.split("#")[0]
            img_paths.append(os.path.join(IMG_BASE_DIR, img_name))
            caps.append(cap)
    return img_paths, caps

# ---------- DATASETS ---------------------------------------------
class UniqueImageDataset(Dataset):
    """Iterates over *unique* image paths; returns (index, PIL_image)."""
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):             return len(self.paths)
    def __getitem__(self, idx):    return idx, safe_image(self.paths[idx])

def collate_img(batch):
    """Batch = list[(idx, PIL)].  Returns (idx_tensor, list[PIL])."""
    idxs, imgs = zip(*batch)
    return torch.tensor(idxs, dtype=torch.long), list(imgs)

class CaptionDataset(Dataset):
    def __init__(self, captions):  self.caps = captions
    def __len__(self):             return len(self.caps)
    def __getitem__(self, i):      return self.caps[i]
def collate_txt(batch):            return list(batch)          # list[str]

# ---------- CLIP --------------------------------------------------
def get_clip():
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE)
    model.eval()
    return proc, model

# ---------- GAUSSIAN ESTIMATOR -----------------------------------
class GaussianEstimator:
    def __init__(self, D, reg=5e-4, device=DEVICE):
        self.D, self.reg, self.device = D, reg, device
        self.count  = 0
        self.sum_x  = torch.zeros(D,      dtype=torch.float64, device=device)
        self.sum_y  = torch.zeros(D,      dtype=torch.float64, device=device)
        self.sum_z  = torch.zeros(2*D,    dtype=torch.float64, device=device)
        self.sum_xx = torch.zeros(D, D,   dtype=torch.float64, device=device)
        self.sum_yy = torch.zeros(D, D,   dtype=torch.float64, device=device)
        self.sum_zz = torch.zeros(2*D,2*D,dtype=torch.float64, device=device)
    def update(self, x, y):
        x, y = x.double().to(self.device), y.double().to(self.device)
        z    = torch.cat([x, y], dim=1)
        self.count += x.size(0)
        self.sum_x  += x.sum(0);   self.sum_y  += y.sum(0);   self.sum_z  += z.sum(0)
        self.sum_xx += x.T @ x;    self.sum_yy += y.T @ y;    self.sum_zz += z.T @ z
    def finalize(self):
        N   = self.count
        μx  = self.sum_x / N;  μy = self.sum_y / N;  μz = self.sum_z / N
        Σx  = (self.sum_xx - N*μx.outer(μx))/(N-1)
        Σy  = (self.sum_yy - N*μy.outer(μy))/(N-1)
        Σz  = (self.sum_zz - N*μz.outer(μz))/(N-1)
        I   = torch.eye(self.D,device=self.device, dtype=torch.float64)
        Σx += self.reg*I
        Σy += self.reg*I
        Σz += self.reg*torch.eye(2*self.D, device=self.device, dtype=torch.float64)
        invΣx = torch.linalg.inv(Σx)
        invΣy = torch.linalg.inv(Σy)
        invΣz = torch.linalg.inv(Σz)
        ld_x  = torch.linalg.slogdet(Σx)[1];  ld_y = torch.linalg.slogdet(Σy)[1];  ld_z = torch.linalg.slogdet(Σz)[1]
        return dict(mu_x=μx, mu_y=μy, mu_z=μz,
                    invΣ_x=invΣx, invΣ_y=invΣy, invΣ_z=invΣz,
                    logdet_x=ld_x, logdet_y=ld_y, logdet_z=ld_z)

# ---------- MAIN --------------------------------------------------
def main(samples_json_path):
    t0 = time.time()
    with open(samples_json_path, 'r') as f:
        all_samples = json.load(f)
    # caps40k = ["A photo depicts "+ c for c in caps40k]
    for model_name, samples in tqdm(all_samples.items(), desc="Models"):
        # preds = [s["generated_caption"] for s in samples][:save_samples]
        paths = [s["image_path"] for s in samples]
        captions = [s["base_caption"]  for s in samples]

        uniq_paths = list(dict.fromkeys(paths))    # preserves first‑seen order
        # maps image‑path → index in unique list
        path2idx = {p:i for i,p in enumerate(uniq_paths)}

        # 1) CLIP setup
        proc, model = get_clip()

        # 2) --------- IMAGE FEATURES (stream, no memory blow‑up) -------
        img_bank = torch.empty(len(uniq_paths), 768, dtype=torch.float64)   # on CPU
        img_ds   = UniqueImageDataset(uniq_paths)
        img_dl   = DataLoader(img_ds, batch_size=IMG_BS, shuffle=False,
                              num_workers=NUM_IMG_W, pin_memory=PIN_MEMORY,
                              collate_fn=collate_img)

        print("🖼  Encoding images …")
        with torch.inference_mode():
            for batch_idx, (idxs, imgs) in enumerate(tqdm(img_dl)):
                inputs = proc(text=["_"]*len(imgs), images=imgs,
                              return_tensors="pt", padding=True)
                raw_feats = model.get_image_features(inputs["pixel_values"].to(DEVICE)).cpu()
                # ℓ₂-normalize each row before storing
                feats = torch.nn.functional.normalize(raw_feats, p=2, dim=-1).double()
                img_bank[idxs] = feats      # store
                del inputs, feats
                torch.cuda.empty_cache()          # optional – forces allocator to return unused blocks

        
            # 3) --------- TEXT  FEATURES ----------------------------------
        txt_feats_all = torch.empty(len(captions), 768, dtype=torch.float64)
        txt_ds = CaptionDataset(captions)
        txt_dl = DataLoader(txt_ds, batch_size=TXT_BS, shuffle=False,
                            num_workers=0, collate_fn=collate_txt)   # strings = cheap

        print("✏️  Encoding captions …")
        start = 0

        with torch.inference_mode():
            for texts in tqdm(txt_dl):
                B = len(texts)
                dummy_imgs = [Image.new("RGB", (224,224))] * B
                inputs = proc(text=texts, images=dummy_imgs,
                              return_tensors="pt", padding=True)
                raw_f = model.get_text_features(
                        inputs["input_ids"].to(DEVICE),
                        attention_mask=inputs["attention_mask"].to(DEVICE)).cpu()
                f = torch.nn.functional.normalize(raw_f, p=2, dim=-1).double()
                txt_feats_all[start:start+B] = f
                start += B

        
        # 4) --------- MAP 8k → 40k ------------------------------------
        # img_feats_all = img_bank[[path2idx[p] for p in paths40k]]  
        img_feats_all = img_bank

        # img_mean = img_feats_all.mean(dim=0, keepdim=True)
        # txt_mean = txt_feats_all.mean(dim=0, keepdim=True)
        
        # # subtract so each feature now has zero mean
        # img_feats_all = img_feats_all - img_mean
        # txt_feats_all = txt_feats_all - txt_mean

        if SAVE_TENSORS:
            torch.save(img_feats_all, "image_feats_40k.pt")
            torch.save(txt_feats_all, "text_feats_40k.pt")
            print("✓ Saved feature tensors to disk")

        
        # 5) --------- GAUSSIAN FIT ------------------------------------
        est = GaussianEstimator(D=768, reg=5e-4)  
        vec_dl = DataLoader(
            TensorDataset(img_feats_all, txt_feats_all),
            batch_size=EST_BS,
            shuffle=False,
            num_workers=2,
            pin_memory=PIN_MEMORY
        )
        for xb, yb in tqdm(vec_dl, desc="Estimator"):
            est.update(xb, yb)
        dists = est.finalize()
        print("Finished – saw", est.count, "samples.  logdet_x:", float(dists["logdet_x"]))
        print(f"Total wall-time: {time.time()-t0:.1f}s")
        save_gaussian(dists, f"mid_gaussian_coco")
        break

if __name__ == "__main__":
    main("output_coco.json")


