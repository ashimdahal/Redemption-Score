import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
import os
import json
from tqdm import tqdm
import timm  # for DINO

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prefix_pred = "/kaggle/input/flickr-pred/flickr_pred"
    prefix_ref  = "/kaggle/input/flickr/flickr"
    prefix_base = "/kaggle/input/flickr8k/Flickr8k_Dataset"
    prediction_caption_image = [os.path.join(prefix_pred, e["prediction_caption_image"]) for e in data]
    reference_caption_image  = [os.path.join(prefix_ref,  e["reference_caption_image"]) for e in data]
    base_image               = [os.path.join(prefix_base, e["base_image"])              for e in data]
    return prediction_caption_image, reference_caption_image, base_image

class IndexMatchedImageDataset(Dataset):
    def __init__(self, image_list1, image_list2, image_list3, transform=None):
        min_length = min(len(image_list1), len(image_list2), len(image_list3))
        self.image_list1 = image_list1[:min_length]
        self.image_list2 = image_list2[:min_length]
        self.image_list3 = image_list3[:min_length]
        self.comparison_pairs = []
        for idx in tqdm(range(min_length), desc="Generating comparison pairs"):
            self.comparison_pairs += [
                {'img1_path': self.image_list1[idx], 'img2_path': self.image_list2[idx], 'comparison_type': '1_2', 'image_idx': idx},
                {'img1_path': self.image_list2[idx], 'img2_path': self.image_list3[idx], 'comparison_type': '2_3', 'image_idx': idx},
                {'img1_path': self.image_list3[idx], 'img2_path': self.image_list1[idx], 'comparison_type': '3_1', 'image_idx': idx},
            ]
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), # DINO models typically use ImageNet normalization, consider mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ])

    def __len__(self):
        return len(self.comparison_pairs)

    def __getitem__(self, i):
        pair = self.comparison_pairs[i]
        img1 = Image.open(pair['img1_path']).convert('RGB')
        img2 = Image.open(pair['img2_path']).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return {
            'img1': img1,
            'img2': img2,
            'img1_path': pair['img1_path'],
            'img2_path': pair['img2_path'],
            'comparison_type': pair['comparison_type'],
            'image_idx': pair['image_idx']
        }

def compute_dino_similarities(dataloader, device='cuda'):
    model = timm.create_model('timm/vit_base_patch8_224.dino', pretrained=True)
    model.eval().to(device)

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing DINO similarities"):
            x1 = batch['img1'].to(device)
            x2 = batch['img2'].to(device)
            
            e1 = model.forward_features(x1) 
            e2 = model.forward_features(x2)
            
            e1_cls = e1[:, 0] 
            e2_cls = e2[:, 0] 

            e1_cls = F.normalize(e1_cls, dim=1)
            e2_cls = F.normalize(e2_cls, dim=1)
            
            sims = torch.sum(e1_cls * e2_cls, dim=1) 
            
            for i, sim in enumerate(sims.cpu().tolist()):
                results.append({
                    'img1_path': batch['img1_path'][i],
                    'img2_path': batch['img2_path'][i],
                    'comparison_type': batch['comparison_type'][i],
                    'image_idx': batch['image_idx'][i].item() if torch.is_tensor(batch['image_idx'][i]) else batch['image_idx'][i], # Ensure it's a number
                    'dino_similarity': sim 
                })
    return results

def organize_results_by_index(results):
    organized_results = {}
    for r in tqdm(results, desc="Organizing DINO results"):
        idx = r['image_idx']
        ct  = r['comparison_type']
        organized_results.setdefault(idx, {})[ct] = r['dino_similarity'] 
    return organized_results

def save_results_to_csv(organized_results, output_file):
    """
    Save the results to a CSV file with three scores per line,
    matching the LPIPS script's format.

    Args:
        organized_results: Dictionary mapping image index to scores by comparison type
        output_file: Path to output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['list1_list2', 'list2_list3', 'list3_list1'])

        for image_idx in tqdm(sorted(organized_results.keys()), desc="Saving DINO results to CSV"):
            result_for_index = organized_results[image_idx]

            score_1_2 = result_for_index.get('1_2', 0.0)
            score_2_3 = result_for_index.get('2_3', 0.0)
            score_3_1 = result_for_index.get('3_1', 0.0)

            writer.writerow([score_1_2, score_2_3, score_3_1])

def main(samples_json_path):
    with open(samples_json_path, 'r') as f:
        all_samples = json.load(f)
    # caps40k = ["A photo depicts "+ c for c in caps40k]
    for model_name, samples in tqdm(all_samples.items(), desc="Models"):
        # preds = [s["generated_caption"] for s in samples][:save_samples]
        base = [s["image_path"] for s in samples][:3000]
        ref = ["model_comparison_batch_sdx1" +"/"+ "references" + "/conceptual_captions/" +b.split('/')[1].split('.')[0]+"_ref.png" for b in base][:3000]
        pred = ["model_comparison_batch_sdx1" +"/"+ "predictions" + "/" +model_name +"/"+ model_name +"_"+ b.split('/')[1].split('.')[0]+"_pred.png" for b in base][:3000]

        # pred, ref, base = load_data(json_path)

        dataset = IndexMatchedImageDataset(pred, ref, base)
        print(f"Dataset size: {len(dataset)} comparisons")

        dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True) # Consider adjusting batch_size based on GPU memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        results = compute_dino_similarities(dataloader, device)
        organized = organize_results_by_index(results)
        
        output_csv_file = f"dino_scores_{model_name}.csv"
        save_results_to_csv(organized, output_csv_file)
        print(f"Saved DINO scores to {output_csv_file}")

        print(f"\nSample of {output_csv_file} content (first 3 rows):")
        try:
            with open(output_csv_file, 'r') as f:
                for i in range(3): 
                    line = f.readline()
                    if not line:
                        break
                    print(line.strip())
        except FileNotFoundError:
            print(f"Error: {output_csv_file} not found.")


if __name__ == "__main__":
    main("output.json")
