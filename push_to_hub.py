from huggingface_hub import HfApi, create_repo
import os
import glob

# Configuration
ORG_NAME = "ashimdahal"
HF_TOKEN = open("./hf_token.txt").read().strip()

# List of files we want to upload (adapter files only)
ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",  # Alternative to safetensors
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "trainer_state.json"
]

def upload_all_models():
    api = HfApi()
    
    # Process each model in results/
    for model_dir in os.listdir("./results"):
        model_path = os.path.join("./results", model_dir)
        if not os.path.isdir(model_path):
            continue

        # Create repo
        repo_id = f"{ORG_NAME}/{model_dir}"
        create_repo(repo_id, exist_ok=True, token=HF_TOKEN, private=True)
        
        # Get processor name from directory name
        processor_name, decoder_name = parse_components(model_dir)

        uploaded_files = upload_peft_model_files(model_path, model_dir, api, repo_id, processor_name)
        tensorboard_exists = upload_training_artifacts(api, repo_id, processor_name)
        upload_model_card(api, repo_id, processor_name, decoder_name)
        
        print(f"\nSummary for {model_dir}:")
        print(f"Uploaded files: {', '.join(uploaded_files)}")
        print(f"TensorBoard uploaded: {tensorboard_exists}")
        print(f"Missing files: {', '.join(set(ADAPTER_FILES) - uploaded_files)}\n")
        
def upload_peft_model_files(model_path, model_dir, api, repo_id, processor_name):
    # Track which files we successfully uploaded
    uploaded_files = set()
    
    # Selectively upload only adapter files from results folder
    for file in ADAPTER_FILES:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file,
                repo_id=repo_id,
                commit_message=f"Adding {file} from results folder",
                token=HF_TOKEN
            )
            uploaded_files.add(file)
            print(f"Uploaded {file} from results folder")
    
    # Check if we need to find missing files from the checkpoint
    missing_files = set(ADAPTER_FILES) - uploaded_files
    if missing_files and ("adapter_model.safetensors" in missing_files or 
                          "adapter_model.bin" in missing_files or
                          "training_args.bin" in missing_files or 
                          "trainer_state.json" in missing_files):
        # Find the highest checkpoint in trainer_logs
        checkpoint_dir = find_highest_checkpoint(processor_name)
        
        if checkpoint_dir:
            print(f"Found checkpoint: {checkpoint_dir}")
            # Check for missing files in the checkpoint
            for file in missing_files:
                file_path = os.path.join(checkpoint_dir, file)
                if os.path.exists(file_path):
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=file,
                        repo_id=repo_id,
                        commit_message=f"Adding {file} from checkpoint",
                        token=HF_TOKEN
                    )
                    uploaded_files.add(file)
                    print(f"Uploaded {file} from checkpoint")
            
            # Special case: if adapter_model.safetensors is missing, try pytorch_model.bin
            if "adapter_model.safetensors" in missing_files and "adapter_model.bin" not in uploaded_files:
                pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                if os.path.exists(pytorch_path):
                    api.upload_file(
                        path_or_fileobj=pytorch_path,
                        path_in_repo="pytorch_model.bin",
                        repo_id=repo_id,
                        commit_message="Adding pytorch_model.bin from checkpoint",
                        token=HF_TOKEN
                    )
                    print(f"Uploaded pytorch_model.bin from checkpoint")
    
    # Provide a summary for this model
    return uploaded_files

def upload_training_artifacts(api, repo_id, processor_name):
    # Upload TensorBoard logs
    log_path = os.path.join("logs_tf_board", processor_name)
    if os.path.exists(log_path):
        api.upload_folder(
            folder_path=log_path,
            repo_id=repo_id,
            path_in_repo="runs",
            commit_message="Adding training logs",
            token=HF_TOKEN,
        )
        return True
    return False

def upload_model_card(api, repo_id, processor_name, decoder_name):
    card_content = f"""
---
tags:
- multimodal
- vision 
- nlp
- image-to-text
base_model:
- {decoder_name}
license: apache-2.0
---
# Introduction
This model is a qlora adapter for the combination of {processor_name} + {decoder_name} fine tuned by @ashimdahal for image captioning.

We trained a peft adapter for the model based on the project: 
https://github.com/ashimdahal/captioning_image/blob/main

### More model specific docs coming soon.
"""
    readme_path = f"README_{processor_name.replace('/','-')}.md"

    with open(readme_path, "w") as f:
        f.write(card_content)

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=HF_TOKEN
    )

def parse_components(model_dir):
    try:
        processor_part, decoder_part = model_dir.split("_", 1)
        processor_part = processor_part.split("-")[0] + "/" + "-".join(processor_part.split("-")[1:])
        decoder_part = decoder_part.split("-")[0] + "/" + "-".join(decoder_part.split("-")[1:])
        return processor_part, decoder_part
    except ValueError:
        raise ValueError(f"Invalid model directory format: {model_dir}")

def find_highest_checkpoint(processor_name):
    """Find the highest checkpoint in trainer_logs for the given processor"""
    checkpoint_base = os.path.join("trainer_logs", processor_name)
    if not os.path.exists(checkpoint_base):
        print(f"No trainer logs found for {processor_name}")
        return None
    
    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_base, "checkpoint-*"))
    
    if not checkpoint_dirs:
        print(f"No checkpoints found for {processor_name}")
        return None
    
    # Sort by checkpoint number
    checkpoints = sorted(
        checkpoint_dirs,
        key=lambda x: int(x.split("-")[-1])
    )
    
    # Return the highest checkpoint
    return checkpoints[-1]

if __name__ == "__main__":
    upload_all_models()
    print("All models uploaded successfully!")
