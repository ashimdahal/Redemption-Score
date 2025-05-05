
# Import standard Python libraries for file operations, data structures, error handling, etc.
import json                     # For reading model configurations and writing results
import random                   # For selecting random samples to save
import glob                     # For finding files (though not explicitly used later, maybe intended)
import albumentations as A      # For image transformations/augmentations
import os                       # For environment variables (like TOKENIZERS_PARALLELISM)
import traceback                # For printing detailed error stack traces
import sys                      # For system-specific parameters and functions (like stderr)
from pathlib import Path        # For object-oriented filesystem paths

# Import core machine learning and data handling libraries
import torch                    # PyTorch library for tensors and neural networks
from torch.utils.data import DataLoader, Dataset # PyTorch utility for loading data in batches
import numpy as np              # NumPy for numerical operations (especially in metrics)
from tqdm import tqdm           # Library for displaying progress bars during loops
from PIL import Image           # Python Imaging Library for opening and manipulating images

# Import necessary classes from the Hugging Face transformers library
from transformers import (
    AutoTokenizer,              # Automatically loads the correct tokenizer based on model name
    AutoProcessor,              # Automatically loads the correct processor (often combines tokenizer and feature extractor)
    AutoModelForCausalLM,       # Base class for loading causal language models (e.g., GPT-2, Llama)
    AutoModelForSeq2SeqLM,      # Base class for loading sequence-to-sequence models (e.g., T5, BART)
    BlipForConditionalGeneration, # Specific model class for BLIP image captioning
    Blip2ForConditionalGeneration,# Specific model class for BLIP-2
    BartForConditionalGeneration, # Specific model class for BART (if used as a decoder)
    GPT2LMHeadModel,            # Specific model class for GPT-2 (if used as a decoder)
    T5ForConditionalGeneration, # Specific model class for T5 (if used as a decoder)
    LlamaForCausalLM,           # Specific model class for Llama (if used as a decoder)
    Qwen2VLForConditionalGeneration, # Specific model class for Qwen-VL V2
    Qwen2_5_VLForConditionalGeneration,# Specific model class for Qwen-VL V2.5 (Placeholder if exists)
    VisionEncoderDecoderModel,  # Wrapper model combining a vision encoder and a text decoder
    Pix2StructForConditionalGeneration, # Specific model class for Pix2Struct (e.g., for UI screenshots)
    GitForCausalLM,             # Specific model class for GIT image captioning
    ViTImageProcessor,          # Specific processor class for Vision Transformer (ViT) models
    BertLMHeadModel,            # Specific model class for BERT used as a language model (if used as decoder)
    BertModel,                  # Base BERT model class (if used as part of an encoder)
    BitsAndBytesConfig,         # Configuration class for 4-bit/8-bit quantization
    MllamaForConditionalGeneration, # Specific model class for MLLaMA (if defined in transformers or custom)
    # Specific processor classes (often needed for type checking or specific methods)
    BlipProcessor,
    Pix2StructProcessor,
    Qwen2VLProcessor,
    Qwen2_5_VLProcessor,        # Placeholder if exists
    MllamaProcessor,            # Placeholder if exists
    Blip2Processor,
    # DataCollatorWithPadding # No longer needed as we implement custom padding
)
# Import PEFT (Parameter-Efficient Fine-Tuning) components
from peft import PeftModel      # Class for loading PEFT adapters (like LoRA) on top of base models

# Import dataset and metric libraries
from datasets import load_dataset # Function to load datasets from Hugging Face Hub or local files
import evaluate                 # Hugging Face library for easily calculating evaluation metrics

# --- Attempt to import local utilities ---
# These imports suggest the script is part of a larger project structure named 'captioning_image'
try:
    # Import the custom dataset class for Conceptual Captions
    from captioning_image.data import ConceptualCaptionsDataset
    print("Successfully imported ConceptualCaptionsDataset from captioning_image/data.py")
except ImportError as e:
    # Handle error if the custom dataset file isn't found
    print(f"Error importing local module 'data.py': {e}", file=sys.stderr)
    print("Please ensure 'data.py' (containing ConceptualCaptionsDataset) is in the Python path (e.g., inside 'captioning_image').", file=sys.stderr)
    raise # Stop execution if the essential dataset class cannot be imported

try:
    # Import custom model wrapper (Collator is now defined locally)
    from captioning_image.scripts.multimodel_utils import MultimodalModel #, MultimodalCollator
    print("Successfully imported MultimodalModel from captioning_image/scripts/multimodel_utils.py")
except ImportError as e:
    # Handle error if the custom utilities file isn't found
    print(f"Warning: Error importing local module 'multimodel_utils.py': {e}", file=sys.stderr)
    # Set MultimodalModel to None if import fails, handle downstream
    MultimodalModel = None

# --- Configuration ---
# Define file paths and directories using pathlib for better path handling
# Use .resolve() to get the absolute path, which can be helpful in environments like Kaggle/Colab
VALID_DATA_DIR = Path("../../kaggle/input/valid-dataset/valid_dataset").resolve() # Directory with validation images
MODEL_CONFIG_PATH = Path("./captioning_image/models.json").resolve() # Path to the JSON file defining models to evaluate
ADAPTER_RESULTS_DIR = Path("./results/").resolve()      # Default directory to look for locally saved PEFT adapters
EVALUATION_OUTPUT_DIR = Path("./evaluation_results/").resolve() # Directory where evaluation results (metrics, samples) will be saved
HF_CACHE_DIR = "./hf_cache"                             # Optional directory for caching Hugging Face downloads (models, datasets)
DEFAULT_EVAL_BATCH_SIZE = 4                             # Default number of samples per batch during evaluation
NUM_EVAL_SAMPLES = 100                                  # Maximum number of validation samples to process for each model
NUM_SAVE_SAMPLES = 100                                  # Maximum number of prediction/reference pairs to save in the output file
MAX_CAPTION_LENGTH = 128                                # Define a fixed max length for captions/padding

# --- Model Type Definitions ---
# Tuples defining categories of models based on how they should be loaded or handled
# Used later in `select_best_model_inference` for conditional logic
# Models compatible with the VisionEncoderDecoderModel wrapper (though the script later avoids re-wrapping)
VISION_ENCODER_DECODER_COMPATIBLE = (
    BertLMHeadModel,
    GPT2LMHeadModel,
    BertModel
)
# Models identified as needing specific handling or the custom MultimodalModel wrapper
REQUIRES_ORIGINAL_IMPLEMENTATION = (
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Pix2StructForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    MllamaForConditionalGeneration,
)

# --- Helper Function for JSON Serialization ---
def convert_to_serializable(obj):
    """Recursively converts numpy types (int, float, bool, ndarray) in metrics dicts/lists to native Python types for JSON compatibility."""
    if isinstance(obj, dict): # If object is a dictionary
        # Recursively apply conversion to each value
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list): # If object is a list
        # Recursively apply conversion to each item
        return [convert_to_serializable(i) for i in obj]
    # Check for various NumPy integer types
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj) # Convert to standard Python int
    elif isinstance(obj, np.floating): # Check for any NumPy float type
        return float(obj) # Convert to standard Python float
    elif isinstance(obj, (np.ndarray,)): # Check for NumPy array
        return obj.tolist() # Convert to standard Python list
    elif isinstance(obj, np.bool_): # Check for NumPy boolean
        return bool(obj) # Convert to standard Python bool
    elif isinstance(obj, (np.void)): # Check for NumPy void type (sometimes occurs in metrics)
        return None # Convert to None
    return obj # Return object unchanged if not a NumPy type needing conversion

# --- Metrics Calculation Function ---
def calculate_metrics_hf(references, predictions):
    """Calculate standard image captioning metrics using HuggingFace's evaluate library."""
    results = {} # Initialize dictionary to store metric scores
    # Basic check for empty inputs
    if not references or not predictions:
         print("Warning: Empty references or predictions list for metrics.", file=sys.stderr)
         return {"error": "Empty inputs"} # Return error status

    # Prepare inputs for different metrics
    references_single = [str(r) for r in references] # Ensure references are strings (for METEOR, ROUGE, Google_BLEU)
    predictions_clean = [str(p) for p in predictions] # Ensure predictions are strings
    # BLEU expects a list of lists of references (even if only one reference per prediction)
    references_list_for_bleu = [[r] for r in references_single]
    metric_load_errors = [] # Keep track of metrics that failed to load or compute

    # Helper function to load and compute a single metric, handling errors
    def compute_metric(metric_name, load_fn, compute_args):
        try:
            metric = load_fn(metric_name) # Load the metric module (e.g., evaluate.load('bleu'))
            score = metric.compute(**compute_args) # Compute the score using provided arguments
            return score # Return the computed score dictionary
        except ModuleNotFoundError: # Specifically catch errors if a metric's dependency isn't installed
            print(f"Could not load {metric_name}: Dependency not found. Please install required packages (e.g., rouge_score, bert_score).", file=sys.stderr)
            metric_load_errors.append(f"{metric_name} (missing dependency)") # Record the error
            return None # Return None to indicate failure
        except Exception as e: # Catch any other error during loading or computation
            print(f"Could not load or compute {metric_name}: {e}", file=sys.stderr)
            metric_load_errors.append(metric_name) # Record the error
            return None # Return None

    # Define the metrics to compute and their specific arguments
    metrics_to_compute = {
        "BLEU": ("bleu", {"predictions": predictions_clean, "references": references_list_for_bleu}), # Standard BLEU score
        "METEOR": ("meteor", {"predictions": predictions_clean, "references": references_single}), # METEOR score
        "ROUGE": ("rouge", {"predictions": predictions_clean, "references": references_single}),   # ROUGE score (usually includes ROUGE-L)
        "Google_BLEU": ("google_bleu", {"predictions": predictions_clean, "references": references_single}), # Google's BLEU implementation
    }
    # Iterate through the defined metrics and compute them
    for name, (load_name, args) in metrics_to_compute.items():
         score = compute_metric(load_name, evaluate.load, args) # Call the helper function
         if score: results[name] = score # Store the result if computation was successful

    # Try computing BERTScore (can be slow, requires model download)
    try:
        # Determine device for BERTScore calculation (prefer GPU)
        bertscore_device = "cuda" if torch.cuda.is_available() else "cpu"
        # Compute BERTScore
        bertscore_result = compute_metric("bertscore", evaluate.load, {
            "predictions": predictions_clean,
            "references": references_single,
            "lang": "en",                   # Specify language (important for model selection)
            "rescale_with_baseline": True,  # Rescale scores for better interpretation
            "device": bertscore_device      # Specify device
        })
        if bertscore_result: results["BERTScore"] = bertscore_result # Store if successful
    except Exception as e:
         # Handle potential errors during BERTScore calculation
         print(f"BERTScore calculation failed: {e}", file=sys.stderr)
         # Avoid adding duplicate error messages
         if "bertscore (missing dependency)" not in metric_load_errors:
              metric_load_errors.append("BERTScore (calculation error)")

    # If any metrics failed, add an error message to the results
    if metric_load_errors:
        results["metric_errors"] = f"Failed to load/compute: {', '.join(metric_load_errors)}"

    # Convert any NumPy types in the results to standard Python types before returning
    return convert_to_serializable(results)

# --- Custom Collator with Manual Padding ---
class MultimodalCollator:
    """
    Custom data collator that handles multimodal features (image + text).
    It manually pads text sequences to a fixed maximum length.
    """
    def __init__(self, processor, tokenizer, max_length=MAX_CAPTION_LENGTH):
        """
        Initializes the collator.

        Args:
            processor: The processor object (e.g., AutoProcessor instance).
            tokenizer: The tokenizer object (e.g., AutoTokenizer instance).
            max_length (int): The fixed length to pad/truncate text sequences to.
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length # Use the globally defined max length

        # --- Ensure tokenizer has a valid pad_token_id ---
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                print(f"Collator: No pad_token_id found. Setting to eos_token_id ({self.tokenizer.eos_token_id})")
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # Last resort: Force Qwen PAD token if EOS is also missing
                QWEN_END_TOKEN_ID = 151643 # Common ID for Qwen tokenizers
                print(f"Collator: WARNING - No PAD or EOS token ID found. Forcing pad_token_id to {QWEN_END_TOKEN_ID} (Qwen default).")
                self.tokenizer.pad_token_id = QWEN_END_TOKEN_ID
                # Also set the pad_token string if possible
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = "<|endoftext|>" # Qwen's typical EOS/PAD token string

        print(f"Collator initialized. Using pad_token_id: {self.tokenizer.pad_token_id}")
        # --- End pad_token_id setup ---

    def __call__(self, features):
        """
        Processes a list of features (dictionaries) into a batch.

        Args:
            features (list): A list of dictionaries, where each dictionary
                             represents a sample and should contain 'image' and 'text' keys.

        Returns:
            dict: A dictionary containing the batched and processed tensors
                  (e.g., 'pixel_values', 'input_ids', 'attention_mask', 'labels').
                  Returns an empty dictionary if the batch is invalid or empty.
        """
        # --- Debug: Print types at start of call ---
        # print(f"\n--- Collator __call__ ---") # Keep this commented unless actively debugging this part
        # print(f"Processor type: {type(self.processor)}")
        # print(f"Tokenizer type: {type(self.tokenizer)}")
        # --- End Debug ---

        # --- Filter out invalid features ---
        valid_features = []
        valid_texts = [] # Keep track of valid texts separately
        for i, item in enumerate(features):
            img = item.get('image')
            txt = item.get('text')
            # Check for valid image (not None) and valid text (is a non-empty string)
            if img is not None and isinstance(txt, str) and txt.strip():
                valid_features.append(item)
                valid_texts.append(txt) # Add text to list only if feature is valid
            else:
                print(f"Warning: Skipping invalid feature {i}. Image type: {type(img)}, Text type: {type(txt)}, Text: '{txt}'", file=sys.stderr)
        # --- End Filtering ---

        # If no valid features remain after filtering, return an empty batch
        if not valid_features:
            print("Warning: No valid features in batch after filtering!", file=sys.stderr)
            return {}

        # --- Process Images ---
        images = [item["image"] for item in valid_features] # Use only valid features
        pixel_values = None
        flattened_patches = None
        try:
            # --- Refined Image Processing Logic ---
            image_processor_obj = None
            # Case 1: Processor has a dedicated image_processor attribute
            if hasattr(self.processor, 'image_processor') and self.processor.image_processor is not None:
                 # print(f"Collator: Using processor.image_processor ({type(self.processor.image_processor)}) for images.")
                 image_processor_obj = self.processor.image_processor
            # Case 2: Processor *is* an image processor
            elif isinstance(self.processor, ViTImageProcessor):
                 # print(f"Collator: Using processor directly as image processor ({type(self.processor)}).")
                 image_processor_obj = self.processor
            # Case 3: Handle combined processors like Qwen2VLProcessor
            elif isinstance(self.processor, (Qwen2VLProcessor, Qwen2_5_VLProcessor)):
                 # print(f"Collator: Using processor directly for Qwen-VL type ({type(self.processor)}).")
                 # Qwen processor might handle images directly when called without text
                 image_processor_obj = self.processor
            # Fallback/Error
            else:
                 print(f"Collator Warning: Unsure how to extract image processor from {type(self.processor)}. Attempting direct call.", file=sys.stderr)
                 image_processor_obj = self.processor # Might fail

            # Call the determined image processor
            if image_processor_obj is not None and callable(image_processor_obj):
                 # Pass only images, explicitly avoiding text arguments
                 image_inputs = image_processor_obj(images=images, return_tensors="pt")
                 pixel_values = image_inputs.get('pixel_values')
                 flattened_patches = image_inputs.get('flattened_patches') # For Pix2Struct
            else:
                 raise TypeError(f"Could not obtain a callable image processor from {type(self.processor)}")

            # Check if at least one image representation was obtained
            if pixel_values is None and flattened_patches is None:
                 print("Error: Image processing did not yield 'pixel_values' or 'flattened_patches'.", file=sys.stderr)
                 return {}
            # --- End Refined Image Processing Logic ---

        except Exception as e:
            print(f"Error processing images in collator: {e}", file=sys.stderr)
            traceback.print_exc()
            return {} # Return empty on error
        # --- End Image Processing ---

        # --- Tokenize Text and Manually Pad/Truncate ---
        # Use the filtered list of valid texts
        try:
            # print(f"Collator: Tokenizing {len(valid_texts)} valid texts...") # Debug
            # Tokenize without automatic padding/truncation
            # `return_tensors=None` returns lists of IDs
            # Ensure ONLY text is passed here
            text_tokenized = self.tokenizer(text=valid_texts, return_tensors=None, add_special_tokens=True, padding=False, truncation=False)
        except TypeError as te:
             # Catch the specific error if it happens here
             print(f"!!! TypeError during self.tokenizer() call in collator: {te}", file=sys.stderr)
             print(f"    Tokenizer Type: {type(self.tokenizer)}")
             print(f"    Number of texts: {len(valid_texts)}")
             traceback.print_exc()
             return {}
        except Exception as e:
             print(f"Error during initial tokenization in collator: {e}", file=sys.stderr)
             traceback.print_exc()
             return {}


        input_ids_list = text_tokenized['input_ids']
        # Generate attention masks manually if not returned by tokenizer (some don't when padding=False)
        if 'attention_mask' not in text_tokenized:
             attention_mask_list = [[1] * len(ids) for ids in input_ids_list]
        else:
             attention_mask_list = text_tokenized['attention_mask']

        padded_input_ids = []
        padded_attention_mask = []
        pad_token_id = self.tokenizer.pad_token_id # Get the assured pad token ID

        for ids, mask in zip(input_ids_list, attention_mask_list):
            current_length = len(ids)
            if current_length > self.max_length:
                # Truncate
                padded_ids = ids[:self.max_length]
                padded_mask = mask[:self.max_length]
            elif current_length < self.max_length:
                # Pad
                padding_length = self.max_length - current_length
                padded_ids = ids + ([pad_token_id] * padding_length)
                padded_mask = mask + ([0] * padding_length) # 0 for padding in attention mask
            else:
                # No padding/truncation needed
                padded_ids = ids
                padded_mask = mask

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
        # --- End Manual Padding ---

        # --- Convert lists to tensors ---
        try:
            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)
        except Exception as e:
             print(f"Error converting padded sequences to tensors: {e}", file=sys.stderr)
             traceback.print_exc()
             return {}

        # Prepare labels (same as input_ids for captioning, but mask padding tokens)
        labels_tensor = input_ids_tensor.clone()
        labels_tensor[labels_tensor == pad_token_id] = -100 # Use -100 ignore index for loss calculation

        # --- Construct the final batch dictionary ---
        batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor, # Labels used for calculating loss during training/evaluation
        }

        # Add image features based on what was processed
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if flattened_patches is not None:
            batch["flattened_patches"] = flattened_patches
        # Add other potential keys if needed by specific models

        # print(f"--- Collator __call__ End (Batch Keys: {list(batch.keys())}) ---") # Debug
        return batch
# --- End Custom Collator ---


# --- Model Loading Function ---
def select_best_model_inference(config):
    """
    Loads model components based on the provided configuration dictionary `config`.
    Handles quantization, PEFT adapter loading, and applies necessary wrappers.
    Returns the final model, processor, tokenizer, and the compute data type.
    """
    processor = None # Initialize processor variable
    tokenizer = None # Initialize tokenizer variable
    model = None     # Initialize model variable
    # Determine device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configure Quantization (4-bit preferred) ---
    try:
        # Try setting up 4-bit NF4 quantization with BFloat16 compute type
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,              # Enable 4-bit loading
            bnb_4bit_use_double_quant=True, # Use double quantization for better precision
            bnb_4bit_quant_type="nf4",      # Use NF4 data type for weights
            bnb_4bit_compute_dtype=torch.bfloat16 # Use BFloat16 for computations (matrix multiplications)
        )
        print("Using 4-bit quantization config (load_in_4bit=True)")
        quantization_enabled = True # Flag that quantization is active
    except AttributeError:
        # Fallback if 4-bit options are not available (older libraries)
        print("Warning: 'load_in_4bit' not found in BitsAndBytesConfig. Trying 'load_in_8bit=True'. Update transformers/bitsandbytes recommended.", file=sys.stderr)
        try:
            # Try setting up 8-bit quantization
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Using 8-bit quantization config as fallback.")
            quantization_enabled = True
        except AttributeError:
             # Fallback if 8-bit is also not available
             print("Warning: Neither 'load_in_4bit' nor 'load_in_8bit' found. Loading model without quantization.", file=sys.stderr)
             quantization_config = None # No quantization config
             quantization_enabled = False # Flag that quantization is inactive

    # Determine the data type for model computations (weights/activations)
    # Use bfloat16 if 4-bit quantization is active and specifies it, otherwise default to float16
    compute_dtype = torch.bfloat16 if quantization_enabled and hasattr(quantization_config, 'bnb_4bit_compute_dtype') else torch.float16

    # Print loading configuration details
    print(f"\n--- Loading Model Components for: ---")
    print(f"  Processor: {config.get('processor_name', 'N/A')}") # Get processor name from config, default to N/A
    print(f"  Tokenizer: {config.get('tokenizer_name', 'N/A')}") # Added Tokenizer name log
    print(f"  Decoder:    {config.get('decoder_name', 'N/A')}")   # Get decoder model name from config
    adapter_hub_id = config.get("adapter_hub_id")           # Get PEFT adapter Hub ID (optional)
    print(f"  Adapter Hub ID: {adapter_hub_id if adapter_hub_id else 'Not specified'}")
    # Safely determine quantization string for logging
    quant_str = '4-bit' if getattr(quantization_config, 'load_in_4bit', False) else ('8-bit' if getattr(quantization_config, 'load_in_8bit', False) else 'None')
    print(f"  Quantization: {quant_str}")
    print(f"  Compute Dtype: {compute_dtype}")
    print("-" * 35)

    # 1. Load Processor and Tokenizer
    try:
        processor_name = config['processor_name']
        tokenizer_name = config.get("tokenizer_name", processor_name) # Get tokenizer name first

        # --- Explicit Loading for ViT-GPT2 ---
        # Check if the names indicate a ViT-GPT2 style model
        is_vit_gpt2_like = ("vit" in processor_name.lower() and "gpt2" in tokenizer_name.lower()) or \
                           ("ViTImageProcessor" in config.get("processor_class","")) # Check class name if provided

        if is_vit_gpt2_like:
             print(f"Detected ViT-GPT2 like case. Loading ViTImageProcessor for '{processor_name}' and AutoTokenizer for '{tokenizer_name}'.")
             # Explicitly load ViTImageProcessor for images
             processor = ViTImageProcessor.from_pretrained(processor_name)
             # Load the corresponding text tokenizer
             tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) # trust_remote_code likely False for gpt2
        # --- Qwen Logic (Keep as before) ---
        elif "Qwen" in processor_name:
            print(f"Qwen processor detected. Loading AutoProcessor with max_pixels.")
            # Load Qwen processor using AutoProcessor (handles combined logic)
            processor = AutoProcessor.from_pretrained(
                processor_name,
                max_pixels=512*28*28,
                trust_remote_code=True
            )
            # Qwen usually uses the same name for tokenizer, load it separately for clarity
            tokenizer = AutoTokenizer.from_pretrained(processor_name, trust_remote_code=True)
        # --- Fallback to Auto Classes ---
        else:
            print(f"Loading processor '{processor_name}' and tokenizer '{tokenizer_name}' using AutoClasses.")
            # Use AutoProcessor for potentially combined processors (like BLIP)
            processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
            # Load tokenizer separately
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        # --- End Fallback ---

        print(f"Processor loaded: Type={type(processor)}")
        print(f"Tokenizer loaded: Type={type(tokenizer)}")

        # NOTE: PAD token logic moved to MultimodalCollator __init__

    except Exception as e:
        # Handle errors during processor/tokenizer loading
        p_name_err = config.get('processor_name', '<KeyError>')
        t_name_err = config.get('tokenizer_name', '<Defaulting to processor>')
        print(f"Error loading processor/tokenizer for processor='{p_name_err}', tokenizer='{t_name_err}': {e}", file=sys.stderr)
        traceback.print_exc() # Print full traceback for loading errors
        raise ValueError("Failed to load processor or tokenizer.") from e # Re-raise as ValueError

    # 2. Load Base Model (Potentially Quantized)
    try:
        # Get decoder model name (required)
        decoder_name = config['decoder_name']
        # Check if the specified decoder class name exists in the global scope (i.e., was imported)
        if config["decoder_class"] not in globals():
             raise NameError(f"Decoder class '{config['decoder_class']}' not found/imported.")
        # Get the actual class object using eval (use carefully)
        DecoderClass = eval(config["decoder_class"])
        # Prepare keyword arguments for model loading
        load_kwargs = {
             "torch_dtype": compute_dtype,      # Specify the computation data type (float16 or bfloat16)
             "trust_remote_code": True        # Allow loading custom code from Hub if needed
        }
        # Add quantization config only if it's valid and enabled
        if quantization_enabled and quantization_config is not None:
             load_kwargs["quantization_config"] = quantization_config

        # Load the model from pretrained weights using the specified class and arguments
        model = DecoderClass.from_pretrained(decoder_name, **load_kwargs)
        # Set the model to evaluation mode (disables dropout, etc.)
        model.eval()
        print(f"Base model '{decoder_name}' loaded ({quant_str} quantization).")

        # --- Set model config pad_token_id immediately after loading model ---
        print("Setting model config pad_token_id...")
        effective_config = None
        if hasattr(model, 'config'): effective_config = model.config
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'config'): effective_config = model.base_model.config # Handle PEFT wrapping

        if effective_config and tokenizer.pad_token_id is not None:
            if effective_config.pad_token_id != tokenizer.pad_token_id:
                print(f"Updating model config pad_token_id from {effective_config.pad_token_id} to: {tokenizer.pad_token_id}")
                effective_config.pad_token_id = tokenizer.pad_token_id
            else:
                print(f"Model config pad_token_id already matches tokenizer: {tokenizer.pad_token_id}")
        elif effective_config and tokenizer.pad_token_id is None:
             # This case should be handled by the collator init now, but good to log
             print("Warning: Tokenizer pad_token_id is None after loading, cannot set model config pad_token_id.", file=sys.stderr)
        else:
             print("Warning: Could not find config attribute on model or base_model.model to verify/set pad_token_id.", file=sys.stderr)
        # --- End model config pad token setup ---

    except Exception as e:
        # Handle errors during base model loading
        print(f"Error loading base model '{decoder_name}' ({quant_str}): {e}", file=sys.stderr)
        # --- MODIFICATION: Print full traceback ---
        traceback.print_exc()
        # --- END MODIFICATION ---
        raise ValueError(f"Failed to load base decoder model '{decoder_name}'.") from e

    # 3. Load PEFT Adapters (LoRA, etc.) if specified
    peft_loaded = False # Flag to track if adapters were loaded
    adapter_source = None # Store where adapters were loaded from (Hub or local)
    if adapter_hub_id: # Check if an adapter ID was provided in the config
        print(f"Attempting PEFT load from Hub: {adapter_hub_id}")
        try:
            # Load PEFT adapters onto the base model
            # is_trainable=False ensures adapters are loaded in inference mode
            model = PeftModel.from_pretrained(model, adapter_hub_id, is_trainable=False)
            peft_loaded = True
            adapter_source = adapter_hub_id
            print(f"Loaded PEFT adapters from Hub: {adapter_hub_id}")
        except Exception as e:
            # Handle failure to load from Hub, print warning and try local path next
            print(f"Warning: Failed Hub PEFT load ({adapter_hub_id}): {e}. Checking local.", file=sys.stderr)

    # Try loading from local path if Hub loading failed or wasn't specified
    if not peft_loaded:
        # Create a safe directory name based on processor/decoder names
        processor_name_safe = config.get('processor_name','proc').replace('/', '_') # Use .get() and default
        decoder_name_safe = config.get('decoder_name','dec').replace('/', '_')
        # Construct the expected local adapter directory path
        adapter_output_dir = ADAPTER_RESULTS_DIR / f"{processor_name_safe}_{decoder_name_safe}"
        # Check if the directory and the essential adapter_config.json file exist
        if adapter_output_dir.exists() and (adapter_output_dir / "adapter_config.json").exists():
            print(f"Attempting PEFT load from local: {adapter_output_dir}")
            try:
                # Load PEFT adapters from the local directory path
                model = PeftModel.from_pretrained(model, str(adapter_output_dir), is_trainable=False)
                peft_loaded = True
                adapter_source = str(adapter_output_dir)
                print(f"Loaded PEFT adapters from local: {adapter_output_dir}")
            except Exception as e:
                # Handle failure to load from local path
                print(f"Warning: Failed local PEFT load ({adapter_output_dir}): {e}", file=sys.stderr)
        else:
             # Print message if local adapter directory/config doesn't exist
             print(f"No valid adapters found locally at {adapter_output_dir}")

    # If adapters weren't loaded from Hub or local, proceed with the base model
    if not peft_loaded:
        print("Proceeding with base model only.")
    # Ensure model is in eval mode after potential adapter loading
    model.eval()

    # 4. Apply Wrappers / Determine Final Model Structure for Evaluation
    # Get the underlying base model class, accounting for potential PEFT wrapping
    # If peft_loaded, model is PeftModel; access its base_model, then the actual model class instance
    # If not peft_loaded, model is the originally loaded model class instance
    model_to_check = model.base_model.model if peft_loaded else model
    # Start with the current `model` (which might be PeftModel wrapping the base)
    final_model = model

    print(f"Checking type of underlying model: {type(model_to_check)}")

    # Apply logic based on the underlying model's type
    if isinstance(model_to_check, VISION_ENCODER_DECODER_COMPATIBLE):
        # If compatible with standard VED framework (e.g., BERT/GPT2 decoder)
        print(f"Compatible with VED.")
        # The script previously skipped reloading as VED to keep quantization/PEFT.
        # Keep the potentially PEFT-wrapped original model.
        print("Using original loaded model structure (potentially PEFT wrapped).")
        final_model = model # No change needed

    elif MultimodalModel is not None and isinstance(model_to_check, REQUIRES_ORIGINAL_IMPLEMENTATION):
        # If model needs specific handling (e.g., BLIP, Qwen-VL) potentially via MultimodalModel wrapper
        # AND the wrapper was successfully imported
        print(f"Model requires specific handling. Wrapping in MultimodalModel.")
        # Apply the custom MultimodalModel wrapper
        final_model = MultimodalModel(
            processor=processor,
            decoder=model, # Pass the potentially PEFT-wrapped, quantized model as the decoder
            tokenizer=tokenizer
        )
        print("Using MultimodalModel wrapper.")

    elif isinstance(model_to_check, (GitForCausalLM, VisionEncoderDecoderModel, LlamaForCausalLM)):
        # These models are often handled directly by the generation logic later
        print(f"Model type {type(model_to_check)} likely works directly.")
        final_model = model # No change needed

    else:
        # Default case for unhandled model types
        print(f"Model type {type(model_to_check)} not explicitly handled.")
        if MultimodalModel is not None:
             print("Using default MultimodalModel wrapper as fallback.")
             # Apply the custom MultimodalModel wrapper as a default fallback
             final_model = MultimodalModel(
                 processor=processor,
                 decoder=model, # Pass the potentially PEFT-wrapped, quantized model
                 tokenizer=tokenizer
             )
        else:
             print("Warning: MultimodalModel wrapper not available. Using raw model structure.", file=sys.stderr)
             final_model = model # Use the raw model


    # Ensure the final model object is in evaluation mode
    final_model.eval()
    print(f"Final model type for evaluation: {type(final_model)}")
    if peft_loaded: print(f"PEFT adapters loaded from: {adapter_source}")
    # Return the components needed for evaluation
    return final_model, processor, tokenizer, compute_dtype


# --- Evaluation Function ---
# Decorator to disable gradient calculations, saving memory during inference
@torch.no_grad()
def evaluate_model(
    model,                          # The final model object (potentially wrapped/PEFTed)
    processor,                      # The processor instance
    tokenizer,                      # The tokenizer instance
    config,                         # The configuration dictionary for this model
    compute_dtype,                  # The data type used for loading (e.g., torch.bfloat16)
    batch_size=DEFAULT_EVAL_BATCH_SIZE, # Batch size for evaluation
    num_samples=NUM_EVAL_SAMPLES,   # Max number of samples to evaluate
    save_samples=NUM_SAVE_SAMPLES,  # Max number of samples to save details for
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Target device
):
    """Evaluates the given model on the Conceptual Captions dataset and saves results."""
    print(f"\n--- Starting Evaluation: {config.get('decoder_name', 'Unknown Model')} ---")
    print(f"Device: {device}, Batch Size: {batch_size}, Compute Dtype: {compute_dtype}")
    print(f"Eval Samples: {num_samples}, Save Samples: {save_samples}")

    # Safety check if model loading failed earlier and passed None
    if model is None or processor is None or tokenizer is None:
        print("Error: evaluate_model called with model/processor/tokenizer=None. Skipping evaluation.", file=sys.stderr)
        return {"error": "evaluate_model received None for model/processor/tokenizer"} # Return error status

    # 1. Load and Prepare Dataset
    try:
        # Load the dataset metadata (image URLs, captions) from Hugging Face Hub
        # Use 'validation' split as specified
        dataset_hf = load_dataset("google-research-datasets/conceptual_captions", split="validation", cache_dir=HF_CACHE_DIR)
        print(f"Loaded Conceptual Captions validation metadata ({len(dataset_hf)} samples).")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}", file=sys.stderr)
        return {"error": f"Dataset loading failed: {e}"} # Return error status

    # Check if the directory with actual image files exists
    if not VALID_DATA_DIR.exists():
        print(f"Error: Validation image directory not found: {VALID_DATA_DIR}", file=sys.stderr)
        return {"error": f"Validation image directory not found: {VALID_DATA_DIR}"}

    # Find all JPG files in the validation directory
    available_files = list(VALID_DATA_DIR.glob("*.jpg"))
    # Create a map from image index (filename without extension) to file path
    # Only include files where the stem (filename part) is purely digits
    file_map = {int(p.stem): p for p in available_files if p.stem.isdigit()}
    # Get a sorted list of available image indices based on filenames
    available_indices = sorted(list(file_map.keys()))
    if not available_indices:
        print(f"Error: No valid images (numeric .jpg filenames) found in {VALID_DATA_DIR}", file=sys.stderr)
        return {"error": f"No valid images found in {VALID_DATA_DIR}"}
    print(f"Found {len(available_indices)} local images.")

    # Determine the actual number of samples to evaluate based on config and availability
    num_available = len(available_indices)
    actual_num_samples = min(num_samples, num_available) if num_samples is not None and num_samples > 0 else num_available
    # Select the indices to be evaluated (first N available indices)
    eval_indices = available_indices[:actual_num_samples]
    if not eval_indices:
        print("Error: No images available for evaluation after selection.", file=sys.stderr); return {"error": "No images selected"}
    print(f"Selected {len(eval_indices)} samples for evaluation.")

    # Define image transformations using Albumentations
    # Resizes the longest side to 512 pixels, preserving aspect ratio
    transform = A.Compose([A.LongestMaxSize(max_size=512)])
    try:
        # Initialize the custom dataset using the Hugging Face metadata, selected indices, image dir, and transform
        dataset = ConceptualCaptionsDataset(dataset_hf, eval_indices, cache_dir=VALID_DATA_DIR, transform=transform)
        print(f"ConceptualCaptionsDataset initialized with {len(dataset)} samples.")
        # Add a check if dataset length matches expected length
        if len(dataset) == 0 and len(eval_indices) > 0:
             print("Error: Dataset is empty after initialization despite selecting indices.", file=sys.stderr)
             return {"error": "Dataset empty after init"}
    except Exception as e:
         # Handle errors during dataset initialization
         print(f"Error initializing ConceptualCaptionsDataset: {e}", file=sys.stderr); traceback.print_exc();
         return {"error": f"Dataset initialization failed: {e}"}

    # 2. DataLoader
    try:
        # Instantiate the custom data collator (defined locally now)
        collator = MultimodalCollator(processor, tokenizer, max_length=MAX_CAPTION_LENGTH)
        # Create the PyTorch DataLoader
        # shuffle=False: Evaluate in a fixed order
        # collate_fn=collator: Use the custom collator to form batches
        # num_workers=0: Run data loading/collation in the main process (easier for debugging)
        # Set prefetch_factor=None when num_workers=0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0, prefetch_factor=None)
        print(f"DataLoader created with {len(dataloader)} batches.")
    except Exception as e:
        # Handle errors during DataLoader or Collator creation
        print(f"Error creating DataLoader or Collator: {e}", file=sys.stderr); traceback.print_exc();
        return {"error": f"DataLoader creation failed: {e}"}

    # 3. Inference Loop
    try:
        # Move the model to the target device (GPU or CPU)
        model.to(device)
        # Ensure model is in evaluation mode
        model.eval()
    except Exception as e:
        print(f"Error moving model to device {device}: {e}", file=sys.stderr)
        return {"error": f"Failed to move model to device: {e}"}

    # Initialize lists to store all predictions and references
    all_preds = []
    all_references = []
    # Initialize list to store details of samples selected for saving
    samples_to_save = []

    # Determine the number of samples to save details for, ensuring it doesn't exceed evaluated samples
    actual_save_samples = min(save_samples, len(dataset)) if save_samples is not None and save_samples > 0 else len(dataset)
    if actual_save_samples > len(dataset): # Check shouldn't be needed due to min() but safe practice
        print(f"Warning: Requested to save {save_samples} samples, but only {len(dataset)} are being evaluated. Saving all evaluated samples.", file=sys.stderr)
        actual_save_samples = len(dataset) # Adjust down if necessary

    # Get the indices within the *dataset* (0 to len(dataset)-1)
    dataset_indices = list(range(len(dataset)))
    # Randomly select dataset indices for which to save details
    indices_to_save_set = set(random.sample(dataset_indices, actual_save_samples)) if actual_save_samples > 0 else set()
    print(f"Will save details for {len(indices_to_save_set)} samples (indices from evaluated dataset).")


    print(f"Running inference...")
    # Loop through batches provided by the DataLoader
    # Use tqdm for a progress bar
    # @torch.no_grad() context manager disables gradient calculation for the loop
    # with torch.no_grad(): # Already applied decorator to the function
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating Batches")):
        # Check if the collator returned an empty batch (e.g., due to filtering bad data)
        if not batch:
             print(f"Warning: Skipping empty batch {i}", file=sys.stderr)
             continue # Skip to the next batch

        # --- Batch Preparation ---
        batch_on_device = {} # Dictionary to hold batch tensors moved to the target device
        try:
            # Move all tensor items in the batch dictionary to the target device
            for k, v in batch.items():
                 if isinstance(v, torch.Tensor):
                     batch_on_device[k] = v.to(device)
                 # else: keep non-tensors (like potential metadata) on CPU
        except Exception as e:
            # Handle errors during device transfer (e.g., out of memory)
            print(f"\nError moving batch {i} to device {device}: {e}", file=sys.stderr)
            # Skip this batch if moving data fails
            continue

        # Extract labels (ground truth captions) from the batch, remove them from input kwargs
        # .pop() removes the key and returns the value, or None if key not present
        labels = batch_on_device.pop("labels", None)
        # Prepare the keyword arguments for the model's generate method
        # This includes all remaining items in batch_on_device (e.g., pixel_values, input_ids, attention_mask)
        gen_kwargs = batch_on_device #{k: v for k, v in batch_on_device.items() if k != "labels"} # pop already removed labels

        # --- Generation ---
        generated_ids = None # Initialize variable for generated token IDs
        try:
            # Determine the actual model instance to call .generate() on
            # If using PEFT, `model` is PeftModel; its `generate` method handles the adapter.
            # If not using PEFT, `model` is the base model or the MultimodalModel wrapper.
            # The logic below tries to identify the underlying *type* for branching,
            # but calls `.generate()` on the main `model` object passed to the function.

            # --- Determine Underlying Model Type for Logic Branching ---
            # Start with the model passed to the function
            _model_for_type_check = model
            # If it's a PeftModel, get the wrapped base model
            if isinstance(_model_for_type_check, PeftModel):
                 _model_for_type_check = _model_for_type_check.base_model.model
            # If the base model is the custom MultimodalModel wrapper, get its decoder
            is_multimodal_wrapper = False # Flag to check if the *effective* model is the wrapper
            if MultimodalModel is not None and isinstance(_model_for_type_check, MultimodalModel):
                 is_multimodal_wrapper = True
                 model_for_generate = _model_for_type_check.decoder # The object to call generate on is the decoder
            else:
                 model_for_generate = model # Otherwise, call generate on the main model object

            # If *that* decoder/model is ALSO a PeftModel (unlikely but possible), get its base for type checking
            if isinstance(model_for_generate, PeftModel):
                 _model_to_check_type = model_for_generate.base_model.model
            else:
                 _model_to_check_type = model_for_generate
            # Now get the type of the final underlying model class
            decoder_model_type = type(_model_to_check_type)
            # --- End Type Determination ---


            # --- Model-Specific Generation Calls ---
            # Branch based on the underlying model type to use the correct arguments for .generate()
            # NOTE: Always call `model_for_generate.generate()` now

            if decoder_model_type in (MllamaForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration):
                # These models might need input_ids and attention_mask in addition to pixel_values
                # The collator should have prepared these in the `batch` / `gen_kwargs`
                if "input_ids" not in gen_kwargs: raise ValueError(f"Missing 'input_ids' for {decoder_model_type}")
                input_len = gen_kwargs["input_ids"].shape[1] # Get length of input prompt part
                # Pass all relevant kwargs from the batch
                generated_ids = model_for_generate.generate(**gen_kwargs, max_new_tokens=50, do_sample=False)
                # Post-process: remove the input prompt tokens from the generated sequence
                generated_ids = [out_ids[input_len:] for out_ids in generated_ids]

            elif decoder_model_type in (VisionEncoderDecoderModel, GitForCausalLM):
                # These models typically only need pixel_values for image captioning
                if "pixel_values" not in gen_kwargs: raise ValueError(f"Missing 'pixel_values' for {decoder_model_type}")
                # Call generate on the main model object
                generated_ids = model_for_generate.generate(pixel_values=gen_kwargs["pixel_values"], max_new_tokens=50, do_sample=False)

            # Check if the effective model is the MultimodalModel wrapper
            elif is_multimodal_wrapper:
                 args_for_gen = {}
                 # Determine arguments based on the *decoder's* type inside the wrapper
                 if decoder_model_type in (BlipForConditionalGeneration, Blip2ForConditionalGeneration):
                     if "pixel_values" not in gen_kwargs: raise ValueError("Missing 'pixel_values' for BLIP/BLIP2")
                     args_for_gen = {"pixel_values": gen_kwargs["pixel_values"]}
                 elif decoder_model_type is Pix2StructForConditionalGeneration:
                     if "flattened_patches" not in gen_kwargs: raise ValueError("Missing 'flattened_patches' for Pix2Struct")
                     args_for_gen = {"flattened_patches": gen_kwargs["flattened_patches"]}
                 else: # Fallback for other wrapped types
                     print(f"Warning: Using generic generate call (pixel_values) for wrapped model {decoder_model_type}", file=sys.stderr)
                     if "pixel_values" in gen_kwargs: args_for_gen["pixel_values"] = gen_kwargs["pixel_values"]
                     else: raise ValueError(f"Missing image features for wrapped {decoder_model_type}")
                 # --- Call generate on model_for_generate (the decoder) ---
                 generated_ids = model_for_generate.generate(**args_for_gen, max_new_tokens=50, do_sample=False)

            else: # Fallback for any other model type not explicitly handled
                 print(f"Warning: Using generic generate call for model type {decoder_model_type}", file=sys.stderr)
                 # Try common arguments, prefer pixel_values if available
                 if "pixel_values" in gen_kwargs:
                      generated_ids = model_for_generate.generate(pixel_values=gen_kwargs["pixel_values"], max_new_tokens=50, do_sample=False)
                 elif "input_ids" in gen_kwargs: # If it's more like a text model being used strangely
                      generated_ids = model_for_generate.generate(input_ids=gen_kwargs["input_ids"], attention_mask=gen_kwargs.get("attention_mask"), max_new_tokens=50, do_sample=False)
                 else: # No common inputs found
                      raise TypeError(f"Unhandled model type and inputs for generation: {decoder_model_type}")

        except Exception as e:
             # Handle errors during the .generate() call
             print(f"\nError during generation batch {i}: {e}", file=sys.stderr)
             traceback.print_exc() # Print detailed traceback
             # Estimate batch size if possible, otherwise use configured batch size
             num_preds_in_batch = labels.shape[0] if labels is not None else batch_size
             # Create error placeholders for predictions
             preds = ["<GENERATION_ERROR>"] * num_preds_in_batch
             # Try to decode references anyway, or use placeholders
             try:
                 refs = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True) if labels is not None else ["<NO_REFERENCE>"] * num_preds_in_batch
             except:
                 refs = ["<LABEL_DECODE_ERROR>"] * num_preds_in_batch
             # Append error placeholders to the main lists
             all_preds.extend(preds)
             all_references.extend(refs)
             # Skip the rest of the loop for this batch
             continue

        # --- Decode Predictions and References ---
        # Decode references (ground truth labels) into text
        # Handle case where labels might not have been provided by the dataset/collator
        if labels is None:
            refs = ["<NO_REFERENCE>"] * (len(generated_ids) if isinstance(generated_ids, list) else generated_ids.shape[0])
        else:
            try:
                labels_cpu = labels.cpu() # Move labels tensor to CPU
                # Replace -100 (ignore index) with pad token ID for decoding
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 # Use 0 if no pad token
                labels_cpu[labels_cpu == -100] = pad_token_id
                # Decode the label token IDs into strings
                refs = tokenizer.batch_decode(labels_cpu, skip_special_tokens=True)
            except Exception as ref_e:
                 print(f"Error decoding references batch {i}: {ref_e}", file=sys.stderr)
                 refs = ["<LABEL_DECODE_ERROR>"] * (len(generated_ids) if isinstance(generated_ids, list) else generated_ids.shape[0])


        # Decode generated token IDs into predicted text
        try:
            if isinstance(generated_ids, list): # Handle list of output tensors (e.g., from Qwen post-processing)
                 # Ensure items are tensors before moving to CPU
                 valid_ids_cpu = [ids.cpu() for ids in generated_ids if isinstance(ids, torch.Tensor)]
                 # Decode the valid tensors
                 preds = tokenizer.batch_decode(valid_ids_cpu, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 # Handle potential length mismatch if some items weren't tensors
                 expected_len = len(generated_ids)
                 if len(preds) < expected_len:
                     preds.extend(["<DECODING_ERROR>"] * (expected_len - len(preds)))
                 elif len(preds) > expected_len:
                     preds = preds[:expected_len] # Should not happen with batch_decode

            elif isinstance(generated_ids, torch.Tensor): # Handle single tensor output
                 # Decode the tensor
                 preds = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            else: # Handle unexpected type for generated_ids
                 print(f"Error: Unexpected type for generated_ids: {type(generated_ids)}", file=sys.stderr)
                 preds = ["<DECODING_ERROR>"] * len(refs) # Assume length matches refs

            # Ensure preds and refs have the same length after decoding attempts
            if len(preds) != len(refs):
                 print(f"Warning: Length mismatch between predictions ({len(preds)}) and references ({len(refs)}) in batch {i}. Adjusting.", file=sys.stderr)
                 # Adjust lists to be the same length, prioritizing reference length if possible
                 target_len = len(refs)
                 if len(preds) < target_len: preds.extend(["<LENGTH_MISMATCH>"] * (target_len - len(preds)))
                 else: preds = preds[:target_len]

        except Exception as e:
             # Handle errors during prediction decoding
             print(f"\nError decoding predictions batch {i}: {e}", file=sys.stderr)
             preds = ["<DECODING_ERROR>"] * len(refs) # Create error placeholders matching ref length

        # --- Store Results ---
        # Append the decoded predictions and references for this batch to the overall lists
        all_preds.extend(preds)
        all_references.extend(refs)

        # --- Save Selected Samples ---
        # Determine the original dataset indices corresponding to this batch
        batch_start_dataset_idx = i * batch_size # Index offset based on dataloader iteration
        # Iterate through items processed in *this* batch (length determined by decoded preds/refs)
        for j in range(len(preds)):
            # Calculate the corresponding index in the *evaluated dataset*
            # This assumes dataloader processes dataset sequentially
            current_dataset_idx = batch_start_dataset_idx + j
            # Check if this dataset index was selected for saving earlier
            if current_dataset_idx in indices_to_save_set:
                try:
                    # Get the image path using the dataset's method (requires dataset index)
                    # Use current_dataset_idx which refers to the index within the evaluated subset
                    img_path = dataset.get_image_path(current_dataset_idx) # Assumes method exists and takes dataset index
                    # Append details to the list, stripping extra whitespace from text
                    samples_to_save.append({
                        "dataset_index": current_dataset_idx, # Include index for reference
                        "image_path": str(img_path),
                        "reference": refs[j].strip(),
                        "prediction": preds[j].strip()
                        })
                except AttributeError:
                    # Handle cases where the dataset class doesn't have get_image_path
                    print(f"Warning: ConceptualCaptionsDataset does not have 'get_image_path' method. Saving sample without path.", file=sys.stderr)
                    samples_to_save.append({
                        "dataset_index": current_dataset_idx,
                        "reference": refs[j].strip(),
                        "prediction": preds[j].strip()
                        })
                    # Prevent this warning from repeating for every sample
                    indices_to_save_set = set() # Hacky way to stop trying after the first failure
                except IndexError:
                     print(f"Error: Index {current_dataset_idx} out of bounds for dataset (length {len(dataset)}) when saving samples.", file=sys.stderr)
                except Exception as e:
                     # Handle other potential errors during sample saving
                     print(f"Error saving sample details for dataset index {current_dataset_idx}: {e}", file=sys.stderr)

    # --- Post-Inference ---
    # 4. Calculate and Save Metrics
    print("\nCalculating metrics...")
    # Calculate metrics using the collected predictions and references
    metrics = calculate_metrics_hf(all_references, all_preds)
    print("\n--- Evaluation Metrics ---")
    # Print metrics nicely formatted as JSON
    print(json.dumps(metrics, indent=4))

    # 5. Save Results to Files
    # Create a safe directory name from processor/decoder names (replace slashes)
    processor_name_safe = config.get('processor_name', 'proc').replace('/', '_')
    decoder_name_safe = config.get('decoder_name', 'dec').replace('/', '_')
    # Construct the full path for the output subdirectory
    output_subdir = EVALUATION_OUTPUT_DIR / f"{processor_name_safe}_{decoder_name_safe}"
    # Create the directory, including parent directories; ignore error if it already exists
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Define path for the metrics JSON file
    metrics_path = output_subdir / "metrics.json"
    try:
        # Open the file for writing
        with open(metrics_path, "w") as f:
            # Dump the metrics dictionary to the file as JSON with indentation
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_path}")
    except Exception as e:
        # Handle errors during file writing
        print(f"Error saving metrics: {e}", file=sys.stderr)

    # Define path for the samples JSON file
    samples_path = output_subdir / "samples.json"
    try:
        # Open the file for writing
        with open(samples_path, "w") as f:
            # Dump the list of saved samples to the file as JSON with indentation
            json.dump(samples_to_save, f, indent=4)
        print(f"Saved {len(samples_to_save)} samples to: {samples_path}")
    except Exception as e:
        # Handle errors during file writing
        print(f"Error saving samples: {e}", file=sys.stderr)

    print(f"--- Evaluation Finished: {config.get('decoder_name', 'Unknown Model')} ---")
    # Return the computed metrics (optional)
    return metrics


# --- Main Execution Function ---
def main():
    """Loads model configurations from JSON and runs evaluation for each one."""
    # Disable parallelism for Hugging Face tokenizers to avoid potential deadlocks/issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("TOKENIZERS_PARALLELISM set to false.")

    # Check if the model configuration file exists
    if not MODEL_CONFIG_PATH.exists():
        print(f"Error: Config file not found: {MODEL_CONFIG_PATH}", file=sys.stderr); return
    try:
        # Open and load the JSON configuration file
        with open(MODEL_CONFIG_PATH, 'r') as f:
            loaded_data = json.load(f) # Load the raw data first

        # --- Robust Check for Config Structure ---
        model_configs_list = []
        if isinstance(loaded_data, list):
            # Check if it's a list of lists
            if len(loaded_data) > 0 and isinstance(loaded_data[0], list):
                 print("Loaded config is a list containing another list. Using the inner list.")
                 model_configs_list = loaded_data[0]
            else:
                 # Assume it's the expected list of dictionaries
                 print("Loaded config is a list (expected format).")
                 model_configs_list = loaded_data
        elif isinstance(loaded_data, dict):
            # Check for common patterns like {"models": [...]}
            print("Loaded config is a dictionary. Checking for a 'models' key containing the list...")
            if "models" in loaded_data and isinstance(loaded_data["models"], list):
                 model_configs_list = loaded_data["models"]
                 print("Found list under 'models' key.")
            else:
                 print("Warning: Loaded config is a dictionary, but couldn't find a standard key ('models') containing a list of model configs.", file=sys.stderr)
                 raise TypeError("Loaded models.json is a dictionary, but expected structure (e.g., list or {'models': [...]}) not found.")
        else:
            # Unexpected format
            raise TypeError(f"Expected models.json to contain a list or a dictionary with a 'models' key, but got type {type(loaded_data)}")

        if not model_configs_list:
             print("Error: No model configurations found after loading and processing models.json.", file=sys.stderr)
             return

        print(f"Successfully processed {len(model_configs_list)} model configs.")
        # --- End Config Structure Check ---

    except Exception as e:
        # Handle errors reading or parsing the JSON file
        print(f"Error reading/parsing config file {MODEL_CONFIG_PATH}: {e}", file=sys.stderr); return

    # Loop through each model configuration found in the processed list
    for idx, config in enumerate(model_configs_list):
        # --- Add type check inside loop ---
        if not isinstance(config, dict):
            print(f"\n{'!'*20} Configuration Error {'!'*20}\nError: Item at index {idx} in model_configs_list is not a dictionary (type: {type(config)}). Expected a dictionary defining the model.\nSkipping...", file=sys.stderr)
            continue # Skip this iteration
        # --- End type check ---

        # Print separator and progress message
        # Now it's safe to use .get() because we know config is a dict
        print("\n" + "="*100)
        print(f"Processing Model {idx+1}/{len(model_configs_list)}: {config.get('decoder_name', 'N/A')}")
        print("="*100 + "\n")

        # Initialize variables for this model iteration
        model, processor, tokenizer, compute_dtype = None, None, None, None

        # Use a try...finally block to ensure cleanup happens even if errors occur
        try:
            # --- Load Model Components ---
            # Call the function to load model, processor, tokenizer based on the current config
            model, processor, tokenizer, compute_dtype = select_best_model_inference(config)
            # -----------------------------

            # --- Post-Loading Adjustments ---
            # Specific configuration for VisionEncoderDecoderModel instances
            # Check the actual model instance type (handling potential PEFT wrapping)
            _underlying_model = model.base_model.model if isinstance(model, PeftModel) else model
            if isinstance(_underlying_model, VisionEncoderDecoderModel):
                 print("Applying VED specific config adjustments...")
                 # Set the decoder_start_token_id (often BOS or CLS token) which guides generation start
                 start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
                 if start_token_id:
                      # Access config potentially through PeftModel structure
                      effective_config = None
                      if hasattr(model, 'config'): effective_config = model.config
                      elif hasattr(model, 'base_model') and hasattr(model.base_model, 'config'): effective_config = model.base_model.config

                      if effective_config:
                          effective_config.decoder_start_token_id = start_token_id
                          print(f"Set VED decoder_start_token_id to {start_token_id}")
                      else:
                          print("Warning: Could not find config attribute on model or base_model.model to set decoder_start_token_id.")
                 else:
                      print("Warning: Could not determine BOS/CLS token for VED decoder_start_token_id.")

                 # Ensure VED config pad_token_id matches tokenizer if not already set
                 # Access config potentially through PeftModel structure
                 effective_config = None
                 if hasattr(model, 'config'): effective_config = model.config
                 elif hasattr(model, 'base_model') and hasattr(model.base_model, 'config'): effective_config = model.base_model.config

                 if effective_config and effective_config.pad_token_id is None and tokenizer.pad_token_id is not None:
                      effective_config.pad_token_id = tokenizer.pad_token_id
                      print(f"Set VED pad_token_id from tokenizer {tokenizer.pad_token_id}")

            # --- Run Evaluation ---
            # Call the evaluation function with the loaded components and config
            evaluate_model(model=model, processor=processor, tokenizer=tokenizer, config=config,
                           compute_dtype=compute_dtype,
                           batch_size=DEFAULT_EVAL_BATCH_SIZE,
                           num_samples=NUM_EVAL_SAMPLES,
                           save_samples=NUM_SAVE_SAMPLES)

        except ValueError as ve: # Catch configuration/loading errors specifically
             print(f"\n{'!'*20} Configuration Error for {config.get('decoder_name', 'N/A')} {'!'*20}\nError: {ve}\nSkipping...", file=sys.stderr)
        except Exception as e: # Catch any other unexpected error during loading or evaluation
            print(f"\n{'!'*20} Critical Error processing {config.get('decoder_name', 'N/A')} {'!'*20}\nError Type: {type(e).__name__}\nError: {e}", file=sys.stderr)
            traceback.print_exc() # Print the full traceback for debugging
            print(f"{'!'*20} Skipping... {'!'*20}\n")

        finally:
            # --- Resource Cleanup ---
            # This block executes whether the try block succeeded or failed
            print(f"Cleaning up resources for {config.get('decoder_name', 'N/A')}...")
            # Safely delete variables to free memory, checking if they exist first
            if 'model' in locals(): del model
            if 'processor' in locals(): del processor
            if 'tokenizer' in locals(): del tokenizer
            if 'compute_dtype' in locals(): del compute_dtype
            # Explicitly trigger Python's garbage collection
            import gc
            gc.collect()
            # If CUDA (GPU) is available, empty the GPU cache
            if torch.cuda.is_available():
                 torch.cuda.empty_cache(); print("CUDA cache cleared.")
            print("Cleanup complete.")

    # End of loop through model configurations
    print("\n" + "="*100 + "\nAll model configurations processed.\n" + "="*100)

if __name__ == "__main__":
    print("Executing evaluation script...")
    main() # Call the main function to start the process
    print("Evaluation script finished.")


