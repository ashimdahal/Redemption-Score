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
    elif "ertugrul" in name_lower:
        return "Qwen 2VL"
    elif "qwen2.5" in name_lower:
        return "Qwen 2.5 VL"
    elif "microsoft" in name_lower:
        return "GIT"
    elif "nlpconnect" in name_lower:
        return "ViT with GPT2"
    elif "blip2" in name_lower:
        return "BLIP"
    elif "salesforce" in name_lower:
        return "BLIP"
    else:
        return model_name

def process_models(output_dir):
    """Process one random sample from each model"""
    # Output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model directories - FIXED PATH
    evaluate_dir = "evaluation_results"  # updated folder name
    model_dirs = []
    
    # Check if evaluation_results directory exists
    if not os.path.exists(evaluate_dir):
        print(f"Error: {evaluate_dir} directory not found")
        return []
    
    # Get all subdirectories (models)
    for item in os.listdir(evaluate_dir):
        full_path = os.path.join(evaluate_dir, item)
        if os.path.isdir(full_path):
            model_dirs.append(item)
    
    # If no models found, exit
    if not model_dirs:
        print("No model directories found. Exiting.")
        return []
    
    print(f"Found {len(model_dirs)} model directories: {model_dirs}")
    
    # Set up models
    sdxl_pipe, clip_model, clip_processor = setup_models()
    
    results = []
    
    # Process each model
    for model_name in tqdm(model_dirs, desc="Processing models"):
        # Adjust model name based on keywords
        display_name = adjust_model_name(model_name)
        
        sample_file = os.path.join(evaluate_dir, model_name, "samples.json")  # updated file name
        
        # Check if sample file exists
        if not os.path.exists(sample_file):
            print(f"Samples file not found for {model_name}, skipping...")
            continue
        
        # Load samples
        try:
            with open(sample_file, 'r') as f:
                samples = json.load(f)
            
            if not samples:
                print(f"No samples found for {model_name}, skipping...")
                continue
                
            # Pick a random sample
            sample = random.choice(samples)
            
            # Get paths
            image_path = sample["image_path"]
            reference = sample["reference"]
            prediction = sample["prediction"]
            
            # Generate images
            pred_img_path = os.path.join(output_dir, f"{display_name}_pred.png")
            ref_img_path = os.path.join(output_dir, f"{display_name}_ref.png")
            
            # Generate images from captions
            generate_image(sdxl_pipe, prediction, pred_img_path)
            generate_image(sdxl_pipe, reference, ref_img_path)
            
            # Original image path
            base_img_path = os.path.join("valid_dataset", os.path.basename(image_path))
            
            # Calculate similarity if base image exists
            pred_similarity = 0
            ref_similarity = 0
            if os.path.exists(base_img_path):
                pred_similarity = calculate_similarity(clip_model, clip_processor, base_img_path, pred_img_path)
                ref_similarity = calculate_similarity(clip_model, clip_processor, base_img_path, ref_img_path)
            
            # Store results (using the adjusted display name)
            results.append({
                "model": display_name,
                "base_img": base_img_path,
                "pred_img": pred_img_path,
                "ref_img": ref_img_path,
                "reference": reference,
                "prediction": prediction,
                "pred_similarity": pred_similarity,
                "ref_similarity": ref_similarity
            })
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Check if we have results
    if not results:
        print("No results generated. Check your samples.json files.")
        return []
    
    # Create visualization
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
    results = process_models("model_comparison")
    results = process_models("model_comparison_1")
    results = process_models("model_comparison_2")
    if not results:
        print("Script completed but no visualization was created due to errors.")
    else:
        print(f"Successfully processed {len(results)} models.")
