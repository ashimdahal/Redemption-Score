import os
import json
import random
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

def setup_models():
    """Set up SDXL and CLIP models"""
    # Set up SDXL
    sdxl_pipe =StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    sdxl_pipe = sdxl_pipe.to("cuda")
    
    # Set up CLIP for similarity
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    return sdxl_pipe, clip_model, clip_processor

def generate_image(pipe, caption, output_path):
    """Generate image using SDXL"""
    image = pipe(prompt=caption, num_inference_steps=50).images[0]
    image.save(output_path)
    return output_path

def calculate_similarity(clip_model, clip_processor, img1_path, img2_path):
    """Calculate CLIP similarity between two images"""
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        inputs = clip_processor(images=[img1, img2], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            
        img1_features = outputs[0:1] / outputs[0:1].norm(dim=1, keepdim=True)
        img2_features = outputs[1:2] / outputs[1:2].norm(dim=1, keepdim=True)
        
        similarity = torch.nn.functional.cosine_similarity(img1_features, img2_features).item()
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def adjust_model_name(model_name):
    """
    Adjust the model name based on keywords.
    Comparison is done in lowercase.
    """
    name_lower = model_name.lower()
    if "meta" in name_lower:
        return "LLAMA"
    elif "deepseek" in name_lower:
        return "Janus Pro"
    elif "qwen" in name_lower:
        return "Qwen 2VL"
    elif "microsoft" in name_lower:
        return "GIT"
    elif "nlpconnect" in name_lower:
        return "ViT with GPT2"
    elif "nourfakih" in name_lower:
        return "ViT with GPT2"
    elif "salesforce" in name_lower:
        return "BLIP"
    else:
        return model_name

# Modified process_models function to handle sample indices
def process_models(output_dir, sample_index=0):
    """Process specific sample index from each model"""
    os.makedirs(output_dir, exist_ok=True)
    evaluate_dir = "evaluation_results"
    model_dirs = [d for d in os.listdir(evaluate_dir) 
                if os.path.isdir(os.path.join(evaluate_dir, d))]
    
    if not model_dirs:
        return []
    
    sdxl_pipe, clip_model, clip_processor = setup_models()
    results = []
    
    for model_name in tqdm(model_dirs, desc=f"Processing {output_dir}"):
        display_name = adjust_model_name(model_name)
        sample_file = os.path.join(evaluate_dir, model_name, "samples.json")
        
        if not os.path.exists(sample_file):
            continue
            
        with open(sample_file, 'r') as f:
            samples = json.load(f)
            
        if len(samples) <= sample_index:
            continue  # Skip if not enough samples
            
        sample = samples[sample_index]  # Use specific index instead of random
        
        # Generate paths with index
        pred_img_path = os.path.join(output_dir, f"{display_name}_pred_{sample_index}.png")
        ref_img_path = os.path.join(output_dir, f"{display_name}_ref_{sample_index}.png")
        
        # Generate images only if they don't exist
        if not os.path.exists(pred_img_path):
            generate_image(sdxl_pipe, sample["prediction"], pred_img_path)
        if not os.path.exists(ref_img_path):
            generate_image(sdxl_pipe, sample["reference"], ref_img_path)
        
        base_img_path = os.path.join("valid_dataset", os.path.basename(sample["image_path"]))
        
        # Calculate similarities
        pred_similarity = calculate_similarity(clip_model, clip_processor, 
                                              base_img_path, pred_img_path)
        ref_similarity = calculate_similarity(clip_model, clip_processor,
                                             base_img_path, ref_img_path)
        
        results.append({
            "model": display_name,
            "base_img": base_img_path,
            "pred_img": pred_img_path,
            "ref_img": ref_img_path,
            "reference": sample["reference"],
            "prediction": sample["prediction"],
            "pred_similarity": pred_similarity,
            "ref_similarity": ref_similarity
        })
    
    create_visualization(results, output_dir)
    return results

def create_visualization(results, output_dir):
    """Create a simple visualization comparing all models"""
    num_models = len(results)
    
    if num_models == 0:
        print("No results to visualize.")
        return
    
    # Adjust figure size to account for header row
    fig = plt.figure(figsize=(18, 5 * (num_models + 1)))  # +1 for header row
    
    # Create grid with dedicated header row
    gs = gridspec.GridSpec(num_models + 1, 4,  # +1 row for headers
                          height_ratios=[0.3] + [1]*num_models,  # header row height
                          hspace=0.4, wspace=0.1)
    
    # Add header row
    headers = ["Model", "Generated from\nPrediction", "Generated from\nReference", "Original Image"]
    for j, header in enumerate(headers):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(header, fontsize=14, fontweight='bold', pad=0)
        ax.axis('off')

    # Process each model's row
    for i, data in enumerate(results):
        current_row = i + 1  # Start from row 1 after headers

        # Model name column
        ax_model = fig.add_subplot(gs[current_row, 0])
        ax_model.text(0.5, 0.5, data["model"], 
                     ha='center', va='center', fontsize=12,
                     bbox=dict(facecolor='lightblue', alpha=0.3))
        ax_model.axis('off')

        # Helper function for image + caption
        def add_image_subplot(row, col, img_path, caption, similarity):
            ax = fig.add_subplot(gs[row, col])
            try:
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"Similarity: {similarity:.3f}", fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, "Image missing", ha='center', va='center')
            
            # Add wrapped caption
            wrapped = textwrap.fill(caption[:120], width=50)
            ax.text(0.5, -0.15, wrapped, transform=ax.transAxes,
                    ha='center', va='top', fontsize=8)#,bbox=dict(boxstyle='square,pad=1'))
            ax.axis('off')

        # Prediction column
        add_image_subplot(current_row, 1, data["pred_img"], 
                         data["prediction"], data["pred_similarity"])

        # Reference column
        add_image_subplot(current_row, 2, data["ref_img"], 
                         data["reference"], data["ref_similarity"])

        # Original image column
        ax_base = fig.add_subplot(gs[current_row, 3])
        try:
            if os.path.exists(data["base_img"]):
                img = plt.imread(data["base_img"])
                ax_base.imshow(img)
            else:
                ax_base.text(0.5, 0.5, "Image not found", ha='center', va='center')
        except Exception as e:
            ax_base.text(0.5, 0.5, "Load error", ha='center', va='center')
        ax_base.axis('off')

    plt.savefig(os.path.join(output_dir, "model_comparison.png"), 
               dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {os.path.join(output_dir, 'model_comparison.png')}")

# Main execution
if __name__ == "__main__":
    # First collect all available samples per model
    model_samples = {}
    evaluate_dir = "evaluation_results"
    
    # Build sample availability map
    for model_name in os.listdir(evaluate_dir):
        sample_file = os.path.join(evaluate_dir, model_name, "samples.json")
        if os.path.exists(sample_file):
            with open(sample_file, 'r') as f:
                samples = json.load(f)
            model_samples[model_name] = min(len(samples), 3)  # Track available samples
    
    # Generate 3 figures using different samples
    for i in range(3):
        output_dir = f"model_comparison_{i+1}"
        print(f"\n{'='*40}\nGenerating figure {i+1}\n{'='*40}")
        
        # Process models using current index, skipping those without enough samples
        results = []
        for model_name, max_samples in model_samples.items():
            if i < max_samples:
                results += process_models(
                    output_dir=output_dir,
                    sample_index=i
                )
        
        if not results:
            print(f"Figure {i+1} failed to generate")
        else:
            print(f"Created figure {i+1} with {len(results)} models")
