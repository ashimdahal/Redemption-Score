from huggingface_hub import HfApi, create_repo
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoProcessor
import os

# Configuration
ORG_NAME = "ashimdahal"
HF_TOKEN = open("./hf_token.txt").read().strip()

def upload_all_models():
    api = HfApi()
    
    # Process each model in results/
    for model_dir in os.listdir("./results"):
        if not os.path.isdir(os.path.join("./results", model_dir)):
            continue

        # Create repo
        repo_id = f"{ORG_NAME}/{model_dir}"
        create_repo(repo_id, exist_ok=True, token=HF_TOKEN, private=True)
        
        # Get components from directory name
        processor_name, decoder_name = parse_components(model_dir)
        is_peft = check_peft_status(processor_name)

        # Upload main model
        upload_model_files(api, repo_id, model_dir, is_peft)
        
        # Upload training artifacts
        upload_training_artifacts(api, repo_id, processor_name)
        
        # Create model card
        create_model_card(repo_id, processor_name, decoder_name, is_peft)

def parse_components(model_dir):
    try:
        processor_part, decoder_part = model_dir.split("_", 1)
        processor_part = processor_part.split("-")[0] + "/" + "-".join(processor_part.split("-")[1:])
        decoder_part = decoder_part.split("-")[0] + "/" + "-".join(decoder_part.split("-")[1:])
        return processor_part, decoder_part
    except ValueError:
        raise ValueError(f"Invalid model directory format: {model_dir}")

def check_peft_status(processor_name):
    checkpoint_base = os.path.join("trainer_logs", processor_name)
    if not os.path.exists(checkpoint_base):
        return False
    return any("adapter_config.json" in files for _,_,files in os.walk(checkpoint_base))

def upload_model_files(api, repo_id, model_dir, is_peft):
    model_path = os.path.join("./results", model_dir)
    
    # Upload base files
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message="Adding model files",
        token=HF_TOKEN
    )

    # Handle PEFT adapters
    if is_peft:
        upload_peft_artifacts(api, repo_id, model_dir)

def upload_peft_artifacts(api, repo_id, model_dir):
    processor_name = parse_components(model_dir)[0]
    checkpoint_base = os.path.join("trainer_logs", processor_name)
    
    # Find latest checkpoint
    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_base) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1])
    )
    
    if checkpoints:
        # Upload best adapter
        best_checkpoint = os.path.join(checkpoint_base, checkpoints[-1])
        api.upload_folder(
            folder_path=best_checkpoint,
            repo_id=repo_id,
            commit_message="Adding PEFT adapter",
            token=HF_TOKEN,
            path_in_repo="peft_adapter"
        )

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
            # path_in_repo="training_logs"
        )

def create_model_card(repo_id, processor_name, decoder_name, is_peft):
    card_content = f"""
---
tags:
- multimodal
- vision
- nlp
base_model: f{decoder_name}
license: apache-2.0
---

# {processor_name} + {decoder_name} Multimodal Model

## Model Details
**Processor**: [{processor_name}](https://huggingface.co/{processor_name})  
**Decoder**: [{decoder_name}](https://huggingface.co/{decoder_name})  
{'**PEFT Adapter**: Yes – includes both adapter and merged versions.' if is_peft else ''}

## Usage
```python
from transformers import AutoModel, AutoProcessor

{'# For PEFT adapter version:' if is_peft else ''}
{'model = AutoModel.from_pretrained("' + repo_id + '", subfolder="peft_adapter")' if is_peft else 'model = AutoModel.from_pretrained("' + repo_id + '")'}
processor = AutoProcessor.from_pretrained("{repo_id}")

{'# For merged version (PEFT):' if is_peft else ''}
{'model = AutoModel.from_pretrained("' + repo_id + '")' if is_peft else ''}
    """
    # Write the model card to a README.md file and upload it.
    readme_path = f"README_{processor_name.replace('/','-')}.md"
    with open(readme_path, "w") as f:
        f.write(card_content)

    try:
        HfApi().upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"Error uploading model card for {repo_id}: {str(e)}")

if __name__ == "__main__":
    upload_all_models()
    print("All models uploaded successfully!")
