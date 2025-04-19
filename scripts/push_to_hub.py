import os
import glob
import logging
from pathlib import Path
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- Configuration ---
ORG_NAME = "ashimdahal" # Your Hugging Face organization name
RESULTS_DIR = Path("./results") # Directory containing model results (e.g., ./results/Org-Model_Decoder)
TRAINER_LOGS_DIR = Path("./trainer_logs") # Directory containing trainer checkpoints (e.g., ./trainer_logs/Org/Model/checkpoint-xxx)
TF_BOARD_LOGS_DIR = Path("./logs_tf_board") # Directory containing TensorBoard logs (e.g., ./logs_tf_board/Org/Model)
HF_TOKEN_FILE = Path("./hf_token.txt") # File containing your Hugging Face API token
MAKE_REPOS_PRIVATE = True # Set to False to create public repositories

# Files typically associated with PEFT adapters
# *** Removed "README.md" to prevent uploading from results/checkpoint ***
ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin", # Alternative weight format
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "trainer_state.json",
    "config.json", # Base model config often included
    "vocab.json", # Potential vocab file
    "merges.txt", # Potential merges file
    "vocab.txt", # Alternative vocab file
    "added_tokens.json", # Potential added tokens file
    # "README.md", # <-- REMOVED
]

# Files specific to certain full model uploads (like the 'decoder' case in the original script)
# Used if the special 'swin'/'vit-base' condition is met.
# *** Removed "README.md" to prevent uploading from results/checkpoint ***
FULL_MODEL_FILES = [
    "model.safetensors", # Primary full model weights
    "pytorch_model.bin", # Alternative full model weights
    "config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "added_tokens.json",
    # "README.md", # <-- REMOVED
]

# --- Logging Setup ---
# Configure logging to display informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def get_hf_token(token_file: Path) -> str | None:
    """Reads the Hugging Face token from the specified file."""
    try:
        # Read token, removing leading/trailing whitespace
        token = token_file.read_text().strip()
        if not token:
            logging.error(f"Hugging Face token file '{token_file}' is empty.")
            return None
        return token
    except FileNotFoundError:
        logging.error(f"Hugging Face token file not found at: {token_file}")
        return None
    except Exception as e:
        logging.error(f"Error reading token file {token_file}: {e}")
        return None

# *** Rewritten function for robust path finding ***
def get_log_or_checkpoint_base_path(model_dir_name: str, logs_base_dir: Path) -> Path | None:
    """
    Determines the corresponding base path within a log/checkpoint directory structure
    based on the results directory name, handling org names with hyphens (e.g., meta-llama).

    Args:
        model_dir_name: The name of the directory from the 'results' folder
                        (e.g., 'meta-llama-Llama-3.2-11B-Vision-Instruct_some-decoder').
        logs_base_dir: The root directory for logs or checkpoints (e.g., TRAINER_LOGS_DIR).

    Returns:
        The Path object to the specific model's log/checkpoint base directory, or None if not found.

    Logic:
    1. Extracts the part before the first '_' from model_dir_name (key_part).
    2. Lists top-level directories in logs_base_dir (potential orgs).
    3. Checks if key_part starts with '<org_dir>-'.
    4. If yes, constructs path as logs_base_dir / org_dir / rest_of_key_part.
    5. If no org match, checks for a flat structure: logs_base_dir / key_part.
    """
    if not logs_base_dir.is_dir():
        logging.warning(f"Logs base directory '{logs_base_dir}' does not exist.")
        return None

    # Use the part before the first underscore (or the whole name if no underscore)
    key_part = model_dir_name.split('_', 1)[0]

    # --- Try matching Org/Model structure ---
    try:
        # List potential organization directories directly under the logs_base_dir
        potential_org_dirs = [d.name for d in logs_base_dir.iterdir() if d.is_dir()]
    except OSError as e:
        logging.error(f"Could not list directories in '{logs_base_dir}': {e}")
        return None

    found_path = None
    for org_candidate in potential_org_dirs:
        # Check if the key_part starts with the org name followed by a hyphen
        # e.g., does "meta-llama-Llama..." start with "meta-llama-"?
        prefix = f"{org_candidate}-"
        if key_part.startswith(prefix):
            # If it matches, the rest of the key_part is the model name
            model_part = key_part[len(prefix):]
            potential_path = logs_base_dir / org_candidate / model_part
            if potential_path.is_dir():
                logging.debug(f"Found potential log path via org '{org_candidate}': {potential_path}")
                found_path = potential_path
                break # Found the correct path
            else:
                 logging.debug(f"Path {potential_path} does not exist or is not a directory.")


    if found_path:
        return found_path

    # --- Fallback: Check for flat structure (no Org folder) ---
    logging.debug(f"No Org/Model structure found matching '{key_part}'. Checking for flat structure...")
    potential_path_flat = logs_base_dir / key_part
    if potential_path_flat.is_dir():
        logging.debug(f"Found potential flat log path: {potential_path_flat}")
        return potential_path_flat

    # --- If no match found ---
    logging.warning(f"Could not find a matching log/checkpoint directory for key '{key_part}' under '{logs_base_dir}' using Org/Model or flat structure.")
    return None


def find_highest_checkpoint(model_dir_name: str) -> Path | None:
    """
    Finds the path to the checkpoint directory with the highest number
    within the corresponding trainer_logs structure. Uses the robust path finding.
    """
    # Determine the base directory for this model's checkpoints using the improved logic
    checkpoint_base = get_log_or_checkpoint_base_path(model_dir_name, TRAINER_LOGS_DIR)

    if not checkpoint_base or not checkpoint_base.is_dir():
        # get_log_or_checkpoint_base_path already logs warnings/errors
        return None

    # Find all directories matching the 'checkpoint-*' pattern
    try:
        checkpoint_dirs = list(checkpoint_base.glob("checkpoint-*"))
    except OSError as e:
        logging.error(f"Error searching for checkpoints in '{checkpoint_base}': {e}")
        return None


    # Filter out any non-directory items, just in case glob returns files
    checkpoint_dirs = [d for d in checkpoint_dirs if d.is_dir()]

    if not checkpoint_dirs:
        logging.warning(f"No valid checkpoint directories found under {checkpoint_base}")
        return None

    # Sort directories based on the numerical value after 'checkpoint-'
    try:
        checkpoints_sorted = sorted(
            checkpoint_dirs,
            key=lambda p: int(p.name.split("-")[-1]) # Extract number from 'checkpoint-123'
        )
    except (ValueError, IndexError):
        logging.error(f"Could not parse checkpoint numbers in {checkpoint_base}. Found names: {[p.name for p in checkpoint_dirs]}")
        return None

    # The last element in the sorted list is the highest checkpoint
    highest_checkpoint = checkpoints_sorted[-1]
    logging.info(f"Found highest checkpoint for '{model_dir_name}': {highest_checkpoint}")
    return highest_checkpoint

def upload_files_from_location(
    api: HfApi,
    source_dir: Path,
    repo_id: str,
    files_to_check: list[str],
    commit_message_prefix: str,
    hf_token: str
) -> set[str]:
    """
    Uploads files from a given source directory to a specified Hugging Face repository
    if they exist in the source directory.

    Args:
        api: An initialized HfApi instance.
        source_dir: The local directory to search for files.
        repo_id: The Hugging Face repository ID (e.g., 'OrgName/RepoName').
        files_to_check: A list of filenames to check for and upload.
        commit_message_prefix: A prefix for the git commit message (e.g., 'results', 'checkpoint').
        hf_token: The Hugging Face API token.

    Returns:
        A set containing the names of the files that were successfully uploaded.
    """
    uploaded = set()
    if not source_dir.is_dir():
        logging.warning(f"Source directory '{source_dir}' does not exist or is not a directory. Cannot upload files.")
        return uploaded

    for file_name in files_to_check:
        file_path = source_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                logging.info(f"Uploading '{file_name}' from {source_dir.name} to {repo_id}...")
                api.upload_file(
                    path_or_fileobj=str(file_path), # API expects string path
                    path_in_repo=file_name,         # Use the original filename in the repo
                    repo_id=repo_id,
                    commit_message=f"{commit_message_prefix}: Add {file_name}",
                    token=hf_token
                )
                uploaded.add(file_name)
                logging.info(f"Successfully uploaded '{file_name}' from {source_dir.name}")
            except HfHubHTTPError as e:
                 # Specific handling for HTTP errors (like 4xx, 5xx)
                 logging.error(f"HTTP error uploading '{file_name}' from {source_dir.name} to {repo_id}: {e}")
                 # Optionally, add more specific error handling based on status code
                 # if e.response.status_code == 401: logging.error("  Check HF Token permissions.")
            except Exception as e:
                # Catch other potential exceptions during upload
                logging.error(f"Failed to upload '{file_name}' from {source_dir.name} to {repo_id}: {e}")
        # else:
        #     # Optional: Log files that were checked but not found
        #     logging.debug(f"File '{file_name}' not found in {source_dir}")

    return uploaded

# --- Core Upload Functions ---

def upload_model_files(
    api: HfApi,
    model_results_path: Path, # Path to the specific model's results (e.g., ./results/Org-Model_Decoder)
    model_dir_name: str,      # Just the directory name (e.g., Org-Model_Decoder)
    repo_id: str,
    hf_token: str,
    target_files: list[str]   # Which files are we aiming to upload for this repo?
) -> set[str]:
    """
    Uploads model files to a specific repository.

    It prioritizes files found directly in the model's results directory.
    If essential files (like weights or training state) specified in `target_files`
    are missing, it searches the highest corresponding checkpoint directory.

    Args:
        api: An initialized HfApi instance.
        model_results_path: Path object to the model's directory within RESULTS_DIR.
        model_dir_name: The name of the model directory (used for finding checkpoints).
        repo_id: The target Hugging Face repository ID.
        hf_token: The Hugging Face API token.
        target_files: A list of filenames that should ideally be uploaded to this repo.

    Returns:
        A set containing the names of all files successfully uploaded to the repo.
    """
    logging.info(f"--- Processing files for repo: {repo_id} (Source: {model_results_path}) ---")
    logging.info(f"Target files for this repo: {', '.join(target_files)}")

    # 1. Attempt to upload target files found directly in the model's results directory
    uploaded_files = upload_files_from_location(
        api, model_results_path, repo_id, target_files, "results", hf_token
    )
    logging.info(f"Uploaded {len(uploaded_files)} files directly from results directory.")

    # 2. Identify missing files and determine if a checkpoint search is needed
    missing_files = set(target_files) - uploaded_files

    # Define which missing files trigger a checkpoint search (typically weights or training state)
    essential_files_for_checkpoint_search = {
        "adapter_model.safetensors", "adapter_model.bin", # Adapter weights
        "model.safetensors", "pytorch_model.bin",         # Full model weights
        "training_args.bin", "trainer_state.json"          # Training state
    }
    needs_checkpoint_search = any(f in missing_files for f in essential_files_for_checkpoint_search)

    if missing_files and needs_checkpoint_search:
        logging.info(f"Found {len(missing_files)} missing essential files: {', '.join(f for f in missing_files if f in essential_files_for_checkpoint_search)}. Searching checkpoint...")

        # Find the path to the highest numbered checkpoint directory
        checkpoint_dir = find_highest_checkpoint(model_dir_name) # Uses improved path finding

        if checkpoint_dir and checkpoint_dir.is_dir():
            # 3. Attempt to upload the *missing* files from the checkpoint directory
            logging.info(f"Attempting to upload missing files from checkpoint: {checkpoint_dir}")
            uploaded_from_ckpt = upload_files_from_location(
                api, checkpoint_dir, repo_id, list(missing_files), "checkpoint", hf_token
            )
            uploaded_files.update(uploaded_from_ckpt) # Add newly uploaded files to the main set
            logging.info(f"Uploaded {len(uploaded_from_ckpt)} files from checkpoint.")

            # 4. Special Fallback: Check for 'pytorch_model.bin' in checkpoint if adapter weights are *still* missing
            # This assumes 'pytorch_model.bin' in the checkpoint *might* be the adapter weights.
            adapter_weights_were_targets = any(f in target_files for f in ["adapter_model.safetensors", "adapter_model.bin"])
            adapter_weights_still_missing = not any(f in uploaded_files for f in ["adapter_model.safetensors", "adapter_model.bin"])

            if adapter_weights_were_targets and adapter_weights_still_missing:
                 pytorch_model_path_ckpt = checkpoint_dir / "pytorch_model.bin"
                 if pytorch_model_path_ckpt.exists() and pytorch_model_path_ckpt.is_file():
                    logging.info("Adapter weights still missing, attempting to upload 'pytorch_model.bin' from checkpoint as 'adapter_model.bin'...")
                    try:
                        # Decide the target name in the repo. Using 'adapter_model.bin' is often desired for PEFT compatibility.
                        target_name_in_repo = "adapter_model.bin"
                        api.upload_file(
                            path_or_fileobj=str(pytorch_model_path_ckpt),
                            path_in_repo=target_name_in_repo, # Upload as standard adapter name
                            repo_id=repo_id,
                            commit_message=f"checkpoint: Add pytorch_model.bin (as {target_name_in_repo})",
                            token=hf_token
                        )
                        # Ensure we add the name *as uploaded* to the set
                        uploaded_files.add(target_name_in_repo)
                        # If the original target was .safetensors, remove it from missing if we uploaded .bin
                        if "adapter_model.safetensors" in missing_files:
                             missing_files.discard("adapter_model.safetensors")

                        logging.info(f"Uploaded 'pytorch_model.bin' from checkpoint as '{target_name_in_repo}'")
                    except HfHubHTTPError as e:
                         logging.error(f"HTTP error uploading pytorch_model.bin from checkpoint as {target_name_in_repo}: {e}")
                    except Exception as e:
                         logging.error(f"Failed to upload pytorch_model.bin from checkpoint as {target_name_in_repo}: {e}")
                 else:
                     logging.warning("Adapter weights still missing, and 'pytorch_model.bin' not found in checkpoint.")

        else:
            logging.warning(f"Could not find or access checkpoint directory for {model_dir_name} to search for missing files.")
    elif missing_files:
         logging.warning(f"Found {len(missing_files)} missing files, but none triggered a checkpoint search: {', '.join(missing_files)}")

    # Return the complete set of files successfully uploaded for this repo
    return uploaded_files


def upload_training_artifacts(api: HfApi, model_dir_name: str, repo_id: str, hf_token: str) -> bool:
    """
    Uploads the contents of the corresponding TensorBoard log directory to the 'runs'
    subdirectory of the specified repository. Uses the robust path finding.
    """
    # Find the base directory for TensorBoard logs for this model using the improved logic
    log_path_base = get_log_or_checkpoint_base_path(model_dir_name, TF_BOARD_LOGS_DIR)

    if log_path_base and log_path_base.is_dir():
        try:
            logging.info(f"Uploading TensorBoard logs from {log_path_base} to {repo_id}/runs...")
            # Upload the entire folder content to the 'runs' path in the repo
            api.upload_folder(
                folder_path=str(log_path_base),
                repo_id=repo_id,
                path_in_repo="runs", # Standard path for TensorBoard integration on HF Hub
                commit_message="Add training logs",
                token=hf_token,
                # ignore_patterns allows skipping specific files/folders during upload
                ignore_patterns=["*.sagemaker-uploading", "*.sagemaker-uploaded", "checkpoint-*"],
            )
            logging.info(f"Successfully uploaded TensorBoard logs for {repo_id}")
            return True
        except HfHubHTTPError as e:
             logging.error(f"HTTP error uploading TensorBoard logs for {repo_id} from {log_path_base}: {e}")
             return False
        except Exception as e:
            logging.error(f"Failed to upload TensorBoard logs for {repo_id} from {log_path_base}: {e}")
            return False
    else:
        # get_log_or_checkpoint_base_path already logs warnings if path not found
        logging.info(f"Skipping TensorBoard upload for {repo_id} as log directory was not found.")
        return False

def generate_model_card_content(repo_id: str, model_dir_name: str) -> str:
    """
    Generates basic Markdown content for a README.md file (Model Card).
    Includes heuristic guesses for base models based on the directory name.
    Uses the corrected base_model field default.
    """
    # --- Heuristic Base Model Guessing ---
    # Tries to guess processor/decoder from 'results' dir name like 'Proc-Model_Dec-Model'
    # This is imperfect and should be manually verified by the user.
    parts = model_dir_name.split('_', 1)
    processor_key = parts[0]
    decoder_key = parts[1] if len(parts) > 1 else ""

    # Try to format as HF model IDs (e.g., 'Org/ModelName')
    # This guess might still be imperfect for complex names, emphasizing manual verification.
    processor_guess = processor_key.replace("-", "/", 1) if '-' in processor_key else processor_key
    decoder_guess = decoder_key.replace("-", "/", 1) if '-' in decoder_key else decoder_key

    # --- README Content ---
    # Note: The inference example code block is intentionally commented out.
    # Users should uncomment and adapt it.
    content = f"""
---
# Auto-generated fields, verify and update as needed
license: apache-2.0
tags:
- generated-by-script
- peft # Assume PEFT adapter unless explicitly a full model repo
- image-captioning # Add more specific task tags if applicable
base_model: [] # <-- FIXED: Provide empty list as default to satisfy validator
# - {processor_guess} # Heuristic guess for processor, VERIFY MANUALLY
# - {decoder_guess} # Heuristic guess for decoder, VERIFY MANUALLY
---

# Model: {repo_id}

This repository contains model artifacts for a run named `{model_dir_name}`, likely a PEFT adapter.

## Training Source
This model was trained as part of the project/codebase available at:
https://github.com/ashimdahal/captioning_image/blob/main

## Base Model Information (Heuristic)
* **Processor/Vision Encoder (Guessed):** `{processor_guess}`
* **Decoder/Language Model (Guessed):** `{decoder_guess}`

**⚠️ Important:** The `base_model` tag in the metadata above is initially empty. The models listed here are *heuristic guesses* based on the training directory name (`{model_dir_name}`). Please verify these against your training configuration and update the `base_model:` list in the YAML metadata block at the top of this README with the correct Hugging Face model identifiers.

## How to Use (Example with PEFT)

```python
from transformers import AutoProcessor, AutoModelForVision2Seq, Blip2ForConditionalGeneration # Or other relevant classes
from peft import PeftModel, PeftConfig
import torch

# --- Configuration ---
# 1. Specify the EXACT base model identifiers used during training
base_processor_id = "{processor_guess}" # <-- Replace with correct HF ID
base_model_id = "{decoder_guess}" # <-- Replace with correct HF ID (e.g., Salesforce/blip2-opt-2.7b)

# 2. Specify the PEFT adapter repository ID (this repo)
adapter_repo_id = "{repo_id}"

# --- Load Base Model and Processor ---
processor = AutoProcessor.from_pretrained(base_processor_id)

# Load the base model (ensure it matches the type used for training)
# Example for BLIP-2 OPT:
base_model = Blip2ForConditionalGeneration.from_pretrained(
     base_model_id,
     torch_dtype=torch.float16 # Or torch.bfloat16 or float32, match training/inference needs
)
# Or for other model types:
base_model = AutoModelForVision2Seq.from_pretrained(base_model_id, torch_dtype=torch.float16)
base_model = AutoModelForCausalLM
......

# --- Load PEFT Adapter ---
# Load the adapter config and merge the adapter weights into the base model
model = PeftModel.from_pretrained(base_model, adapter_repo_id)
model = model.merge_and_unload() # Merge weights for inference (optional but often recommended)
model.eval() # Set model to evaluation mode

# --- Inference Example ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = ... # Load your image (e.g., using PIL)
text = "a photo of" # Optional prompt start

inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16) # Match model dtype

generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(f"Generated Caption: {{generated_text}}")
```

*More model-specific documentation, evaluation results, and usage examples should be added here.*
"""
    # Need to escape the curly braces within the final print statement for the f-string formatting
    content = content.replace("{generated_text}", "{{generated_text}}")
    return content

def upload_model_card(api: HfApi, repo_id: str, model_dir_name: str, hf_token: str):
    """
    Generates a README.md file with basic model information and usage guidance,
    then uploads it to the specified repository. Overwrites existing README.md.
    """
    readme_content = generate_model_card_content(repo_id, model_dir_name)
    # Use a temporary file path for the generated README content
    # Suffix helps avoid collisions if script runs fast
    temp_readme_path = Path(f"temp_README_{repo_id.replace('/', '_')}_{os.getpid()}.md")

    try:
        # Write the generated content to the temporary file
        temp_readme_path.write_text(readme_content, encoding='utf-8')

        logging.info(f"Uploading generated README.md to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(temp_readme_path),
            path_in_repo="README.md", # Standard filename for model cards
            repo_id=repo_id,
            commit_message="Add/Update generated README.md", # Commit message reflects update
            token=hf_token
        )
        logging.info(f"Successfully uploaded README.md for {repo_id}")

    except HfHubHTTPError as e:
         logging.error(f"HTTP error uploading README.md for {repo_id}: {e}")
         # Add specific check for metadata validation error
         if "Invalid metadata" in str(e):
             logging.error("  >> The generated README.md metadata was rejected by Hugging Face Hub.")
             logging.error(f"  >> Error details: {e}")
    except Exception as e:
        logging.error(f"Failed to generate or upload README.md for {repo_id}: {e}")
    finally:
        # Clean up the temporary file, regardless of success or failure
        if temp_readme_path.exists():
            try:
                temp_readme_path.unlink()
            except OSError as e:
                logging.warning(f"Could not delete temporary README file {temp_readme_path}: {e}")


# --- Main Execution Logic ---

def upload_all_models():
    """
    Main function to iterate through model directories in RESULTS_DIR,
    create Hugging Face repositories, and upload relevant files (adapter files,
    logs, model cards), handling checkpoints and special cases.
    """
    # 1. Get Hugging Face Token
    hf_token = get_hf_token(HF_TOKEN_FILE)
    if not hf_token:
        logging.error("Cannot proceed without a valid Hugging Face token.")
        return

    # 2. Initialize Hugging Face API client
    api = HfApi()

    # 3. Check if the main results directory exists
    if not RESULTS_DIR.is_dir():
        logging.error(f"Results directory not found: {RESULTS_DIR}")
        return

    logging.info(f"Starting model upload process from '{RESULTS_DIR}' to Hugging Face Org: '{ORG_NAME}'")

    # 4. Iterate through each item in the results directory
    for model_path in RESULTS_DIR.iterdir():
        # Skip items that are not directories
        if not model_path.is_dir():
            logging.debug(f"Skipping non-directory item: {model_path.name}")
            continue

        model_dir_name = model_path.name # e.g., Salesforce-blip2-opt-2.7b_Salesforce-blip2-opt-2.7b
        logging.info(f"\n===== Processing model directory: {model_dir_name} =====")

        # --- Upload Main Adapter/Model Repo ---
        # Construct the repository ID based on the directory name
        repo_id = f"{ORG_NAME}/{model_dir_name}"
        logging.info(f"Attempting to create/access main repo: {repo_id} (Private: {MAKE_REPOS_PRIVATE})")

        try:
            # Create the repository on Hugging Face Hub. exist_ok=True prevents errors if it already exists.
            create_repo(repo_id, exist_ok=True, token=hf_token, private=MAKE_REPOS_PRIVATE)
        except Exception as e:
            logging.error(f"Failed to create or access repo {repo_id}: {e}. Skipping this model.")
            continue # Skip to the next model directory if repo creation fails

        # Upload adapter files (or potentially full model if adapter files aren't primary)
        # This function handles finding files in results and checkpoints.
        # *** NOTE: ADAPTER_FILES list no longer includes "README.md" ***
        uploaded_main_files = upload_model_files(
            api=api,
            model_results_path=model_path,
            model_dir_name=model_dir_name,
            repo_id=repo_id,
            hf_token=hf_token,
            target_files=ADAPTER_FILES # Specify we are looking for adapter-related files
        )

        # Upload TensorBoard logs if they exist (uses improved path finding)
        tensorboard_uploaded = upload_training_artifacts(api, model_dir_name, repo_id, hf_token)

        # Generate and upload the definitive README.md file for the repo
        # This is now the *only* place a README.md is uploaded from.
        upload_model_card(api, repo_id, model_dir_name, hf_token)

        # Log summary for the main repository
        logging.info(f"\nSummary for main repo {repo_id}:")
        # Note: README.md might not appear in uploaded_main_files anymore, which is expected.
        logging.info(f"  Uploaded files ({len(uploaded_main_files)}): {', '.join(sorted(list(uploaded_main_files)))}")
        logging.info(f"  TensorBoard uploaded: {tensorboard_uploaded}")
        missing_main_files = set(ADAPTER_FILES) - uploaded_main_files
        # Report potentially missing files based on the target list
        if missing_main_files:
            logging.warning(f"  Potentially missing target adapter files ({len(missing_main_files)}): {', '.join(sorted(list(missing_main_files)))}")
        else:
             logging.info("  All target adapter files accounted for.")


        # --- Special Handling for 'swin' or 'vit-base' (Optional) ---
        # This section mirrors the original script's logic to upload a separate
        # repository possibly containing full model weights for specific model types.
        # Adjust or remove this block based on your actual needs.
        model_dir_name_lower = model_dir_name.lower()
        if "swin" in model_dir_name_lower or "vit-base" in model_dir_name_lower:
            # Define a name for the separate repository (e.g., prefix with 'decoder-' or 'vision_encoder-')
            # Be mindful of potential naming collisions if model_dir_name is very long.
            special_repo_id = f"{ORG_NAME}/vision_encoder-{model_dir_name}"
            logging.info(f"--- Special Handling: Uploading full model components to {special_repo_id} ---")

            try:
                # Create the special repository
                create_repo(special_repo_id, exist_ok=True, token=hf_token, private=MAKE_REPOS_PRIVATE)
            except Exception as e:
                logging.error(f"Failed to create or access special repo {special_repo_id}: {e}")
                # Don't skip the whole script, just this special repo upload
            else:
                # Upload specific files expected for the full model repo using the FULL_MODEL_FILES list
                # *** NOTE: FULL_MODEL_FILES list no longer includes "README.md" ***
                uploaded_special_files = upload_model_files(
                    api=api,
                    model_results_path=model_path,      # Source files might still be in the main results folder
                    model_dir_name=model_dir_name,      # Use original name for finding checkpoints if needed
                    repo_id=special_repo_id,
                    hf_token=hf_token,
                    target_files=FULL_MODEL_FILES # Use the specific list for full models
                )
                # Upload a model card for this special repo too
                # Use a modified name for clarity in the generated card content
                upload_model_card(api, special_repo_id, f"vision_encoder-{model_dir_name}", hf_token)

                # Log summary for the special repository
                logging.info(f"\nSummary for special repo {special_repo_id}:")
                logging.info(f"  Uploaded files ({len(uploaded_special_files)}): {', '.join(sorted(list(uploaded_special_files)))}")
                missing_special_files = set(FULL_MODEL_FILES) - uploaded_special_files
                if missing_special_files:
                     logging.warning(f"  Potentially missing target full model files ({len(missing_special_files)}): {', '.join(sorted(list(missing_special_files)))}")
                else:
                     logging.info("  All target full model files accounted for.")

    logging.info("\n===== Model upload process finished =====")

# --- Script Entry Point ---
if __name__ == "__main__":
    # Execute the main upload function when the script is run directly
    upload_all_models()
    print("\nScript execution finished. Please check the logs above for details on uploads and potential warnings/errors.")



