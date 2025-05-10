import json
import glob
import os
import torch
import gc
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse
import logging
import random
import traceback # Ensure traceback is imported for error logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference_log.txt")
    ]
)
logger = logging.getLogger(__name__)

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    VisionEncoderDecoderModel,
    Pix2StructForConditionalGeneration,
    GitForCausalLM,
    ViTImageProcessor,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    # BitsAndBytesConfig is not used in the user's new load_model_for_inference
    # Assuming these are imported if used by vision_encoder_decoder_compatible
    BertLMHeadModel, 
    GPT2LMHeadModel, 
    BertModel      
)

from peft import PeftModel # PeftConfig is not used in the provided load_model_for_inference
import albumentations as A

# Import custom utilities
from multimodel_utils import MultimodalModel
from janus.models import MultiModalityCausalLM, VLChatProcessor # Kept from user's code
from data import ConceptualCaptionsDataset # Import the custom dataset

# Constants required by the load_model_for_inference function
# These should align with definitions in your environment/training scripts
vision_encoder_decoder_compatible = ( 
    BertLMHeadModel, 
    GPT2LMHeadModel, 
    BertModel      
)
requires_original_implementation = ( 
    BlipForConditionalGeneration,
    Pix2StructForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    MllamaForConditionalGeneration,
    Blip2ForConditionalGeneration, 
    Qwen2_5_VLForConditionalGeneration 
)

GLOBAL_DEVICE_PRIMARY = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- User's provided model loading function ---
def load_model_for_inference(config):
    """
    Load a pretrained model for inference using the adapter hub ID.
    Based on the user's provided working script.
    """
    logger.info(f"Loading model components for: {config.get('decoder_name', 'N/A')}")
    if config.get('adapter_hub_id'):
        logger.info(f"Adapter Hub ID: {config['adapter_hub_id']}")
    else:
        logger.info("No PEFT adapter specified, loading base model only.")

    # Clear CUDA cache before loading each model
    if torch.cuda.is_available():
        logger.debug("Clearing CUDA cache and collecting garbage.")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)  # Give some time for memory to be properly freed
    
    processor = None
    tokenizer = None
    base_model_obj = None # Renamed to avoid confusion with final 'model'

    try:
        # Load the processor
        processor_name = config["processor_name"]
        logger.info(f"Loading processor: {processor_name}")
        if "processor_class" in config and config["processor_class"]:
            ProcessorClass = eval(config["processor_class"]) # Assumes class is imported
            processor = ProcessorClass.from_pretrained(processor_name, trust_remote_code=True)
        else:
            processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
        
        if "Qwen" in processor_name: # Qwen specific processor handling
            logger.info(f"Qwen processor detected. Re-loading with max_pixels.")
            # trust_remote_code=True is important for some AutoProcessor versions with Qwen
            processor = AutoProcessor.from_pretrained(config['processor_name'], max_pixels=512*28*28, trust_remote_code=True)
        
        # Determine device and dtype for base model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_for_base_model = torch.float32 # Default for CPU

        decoder_name = config["decoder_name"]
        DecoderClass = eval(config["decoder_class"]) # Assumes class is imported or globally available
        
        load_kwargs_base = {"trust_remote_code": True}

        if device == "cuda":
            logger.info(f"Target device: CUDA. Loading base model with device_map='auto'.")
            load_kwargs_base["device_map"] = "auto" # Let transformers handle sharding for large models
            try:
                logger.info(f"Attempting to load base model '{decoder_name}' with torch_dtype=torch.bfloat16")
                base_model_obj = DecoderClass.from_pretrained(decoder_name, torch_dtype=torch.bfloat16, **load_kwargs_base)
                dtype_for_base_model = torch.bfloat16
            except Exception as e_bf16:
                logger.warning(f"bfloat16 failed for '{decoder_name}': {e_bf16}. Trying float16.")
                base_model_obj = DecoderClass.from_pretrained(decoder_name, torch_dtype=torch.float16, **load_kwargs_base)
                dtype_for_base_model = torch.float16
        else: # CPU
            logger.info(f"Target device: CPU. Loading base model '{decoder_name}' with torch_dtype=torch.float32")
            base_model_obj = DecoderClass.from_pretrained(decoder_name, torch_dtype=torch.float32, **load_kwargs_base)
            dtype_for_base_model = torch.float32
        
        logger.info(f"Base model '{decoder_name}' loaded with dtype: {dtype_for_base_model}. Initial device distribution: {base_model_obj.hf_device_map if hasattr(base_model_obj, 'hf_device_map') else base_model_obj.device}")

        # Load the adapter if adapter_hub_id is provided
        model_to_return = base_model_obj # Start with base model
        if config.get("adapter_hub_id"):
            adapter_path_or_id = config["adapter_hub_id"]
            # Check if it's a local path first
            if os.path.exists(adapter_path_or_id):
                logger.info(f"Loading PEFT adapter from local path: {adapter_path_or_id}")
            else:
                logger.info(f"Loading PEFT adapter from Hugging Face Hub ID: {adapter_path_or_id}")

            # Important: Load adapter with device_map specified if base_model used device_map,
            # or ensure base_model is on CPU before applying adapter also on CPU then moving.
            # User's code implies direct application. If base_model is on multiple devices via "auto",
            # PeftModel.from_pretrained should handle this.
            try:
                model_to_return = PeftModel.from_pretrained(base_model_obj, adapter_path_or_id)
                logger.info(f"PEFT adapter '{adapter_path_or_id}' loaded and applied.")
            except Exception as e_peft:
                logger.error(f"Failed to load PEFT adapter '{adapter_path_or_id}': {e_peft}")
                logger.warning("Proceeding with base model only due to PEFT loading error.")
                # model_to_return remains base_model_obj
        else:
            logger.info("No 'adapter_hub_id' found in config. Using base model directly.")


        # Ensure the final model is on the target device if not already optimally placed by device_map
        # For models loaded with device_map="auto", their primary device might be cuda:0 or sharded.
        # The .to(device) call here might be redundant or could consolidate a sharded model if small enough.
        # If model is large and sharded by device_map="auto", this .to(device) might cause OOM if 'device' is a single GPU.
        # For now, let's keep it as it was in user's structure, assuming device_map="auto" handles it.
        # If issues, this is a point to revisit.
        # A safer check:
        try:
            current_final_device = next(model_to_return.parameters()).device
            if str(current_final_device) != device and device != "cpu": # Avoid moving if already on a CUDA device from device_map
                 if not hasattr(base_model_obj, 'hf_device_map'): # Only move if not sharded
                    logger.info(f"Moving final model to specified device: {device}")
                    model_to_return = model_to_return.to(device)
            elif str(current_final_device) != device and device == "cpu": # If target is CPU, move it
                logger.info(f"Moving final model to specified device: {device}")
                model_to_return = model_to_return.to(device)

        except StopIteration: # No parameters
            logger.warning("Model has no parameters. Cannot determine or move device.")
        
        model_to_return.eval() 
        
        final_device_check = "unknown"
        try: final_device_check = next(model_to_return.parameters()).device
        except: pass
        logger.info(f"Final model ready. Effective device: {final_device_check}")


        # Load tokenizer
        tokenizer_name = config.get("tokenizer_name", processor_name)
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Set tokenizer pad_token: {tokenizer.pad_token}")
        
        # Handle different model types for potential wrapping (user's logic)
        # Check against the underlying base model if PEFT is applied
        underlying_model_for_check = model_to_return.base_model.model if isinstance(model_to_return, PeftModel) else model_to_return

        if isinstance(underlying_model_for_check, (BlipForConditionalGeneration, Blip2ForConditionalGeneration)):
            logger.info(f"Wrapping {type(underlying_model_for_check).__name__} in MultimodalModel.")
            return MultimodalModel(processor=processor, decoder=model_to_return, tokenizer=tokenizer), processor, tokenizer
        elif isinstance(underlying_model_for_check, (MllamaForConditionalGeneration, GitForCausalLM, VisionEncoderDecoderModel, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)):
            logger.info(f"Using {type(underlying_model_for_check).__name__} directly.")
            return model_to_return, processor, tokenizer
        else: 
            logger.info(f"Defaulting to wrap {type(underlying_model_for_check).__name__} in MultimodalModel.")
            return MultimodalModel(processor=processor, decoder=model_to_return, tokenizer=tokenizer), processor, tokenizer
    
    except Exception as e:
        logger.error(f"Error in load_model_for_inference for config '{config.get('decoder_name', 'N/A')}': {e}")
        traceback.print_exc() # Ensure traceback is printed
        raise

# --- User's provided caption generation function, adapted for PIL image ---
def generate_caption(model, processor, tokenizer, pil_image: Image.Image, image_identifier: str, max_length=50, num_beams=5):
    """
    Generate caption for a single PIL image.
    image_identifier is used for logging purposes.
    """
    try:
        # Determine device from the model itself
        # This ensures inputs are sent to where the model (or its first parameter) resides
        model_device = next(model.parameters()).device
        logger.debug(f"Generating caption for '{image_identifier}' on device: {model_device}")
        
        model.eval() # Ensure model is in eval mode
        
        # pil_image is already a PIL.Image.convert('RGB') object from ConceptualCaptionsDataset
        
        # Gracefully handle different model types (user's logic)
        # Check if the model object (which could be MultimodalModel wrapper) has generate_caption
        if hasattr(model, 'generate_caption') and callable(model.generate_caption):
            logger.debug(f"Using model.generate_caption method for '{image_identifier}'.")
            # Assuming model.generate_caption takes a PIL image
            caption = model.generate_caption(
                pil_image, # Pass PIL image directly
                max_length=max_length,
                num_beams=num_beams
            )
            return caption.strip() if isinstance(caption, str) else caption # Ensure it's a string
        else: # Generic generation logic
            logger.debug(f"Using generic model.generate method for '{image_identifier}'.")
            # User's script uses torch.cuda.amp.autocast.
            # This is generally fine for inference, especially if model is fp16/bf16.
            # It might be redundant if model is already loaded in desired precision and device_map handles it.
            with torch.cuda.amp.autocast(enabled=(model_device.type == "cuda")):  
                inputs = processor(images=pil_image, return_tensors="pt").to(model_device)
                
                with torch.no_grad():
                    if hasattr(model, 'generate'):
                        try:
                            output_ids = model.generate(
                                **inputs,
                                max_length=max_length,
                                num_beams=num_beams,
                                early_stopping=True
                            )
                        except RuntimeError as e_gen: # User's OOM handling
                            if "CUDA out of memory" in str(e_gen):
                                logger.warning(f"CUDA OOM during generation for '{image_identifier}', retrying with reduced beams.")
                                if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
                                output_ids = model.generate(
                                    **inputs,
                                    max_length=max_length,
                                    num_beams=max(1, num_beams // 2), # Reduce beams
                                    early_stopping=True
                                )
                            else:
                                raise # Re-raise other RuntimeErrors
                        
                        # Decoding (user's logic)
                        if hasattr(tokenizer, 'batch_decode'):
                            caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                        elif hasattr(processor, 'decode'): # Some processors might have decode
                            caption = processor.decode(output_ids[0], skip_special_tokens=True)
                        else:
                            logger.error("Neither tokenizer.batch_decode nor processor.decode found.")
                            return "Decoding error"
                        
                        return caption.strip()
                    else:
                        logger.error(f"Model type {type(model)} does not have a 'generate' method.")
                        return "Model doesn't support .generate"
        
    except RuntimeError as e_runtime: # User's OOM fallback to CPU
        if "CUDA out of memory" in str(e_runtime) and model_device.type == "cuda":
            logger.warning(f"CUDA OOM for '{image_identifier}', attempting fallback to CPU for this image.")
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
            
            cpu_device = torch.device("cpu")
            try:
                model.to(cpu_device) # Move model to CPU
                logger.info(f"Moved model to CPU for '{image_identifier}' due to OOM.")
                inputs = processor(images=pil_image, return_tensors="pt").to(cpu_device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_length=max_length, num_beams=max(1, num_beams // 2), early_stopping=True
                    )
                # Move model back to original device if possible (or reload next time)
                # For simplicity, we assume next call to load_model_for_inference will handle device.
                # Or, if processing multiple images for same model, move back after this.
                # model.to(model_device) # This might cause OOM again if not careful.
                # Best to let the main loop handle model reloading/device placement per model config.

                if hasattr(tokenizer, 'batch_decode'):
                    caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                else:
                    caption = processor.decode(output_ids[0], skip_special_tokens=True)
                return caption.strip()
            except Exception as e_cpu_fallback:
                logger.error(f"Error during CPU fallback for '{image_identifier}': {e_cpu_fallback}")
                return f"CPU Fallback Error: {str(e_cpu_fallback)[:100]}..."
        else:
            logger.error(f"RuntimeError generating caption for '{image_identifier}': {e_runtime}")
            traceback.print_exc()
            return f"RuntimeError: {str(e_runtime)[:100]}..."
    except Exception as e_other:
        logger.error(f"General error generating caption for '{image_identifier}': {e_other}")
        traceback.print_exc()
        return f"Error: {str(e_other)[:100]}..."

def main():
    parser = argparse.ArgumentParser(description="Generate captions using trained models")
    parser.add_argument("--test_dir", type=str, default="./valid_dataset/", help="Directory for ConceptualCaptionsDataset cache_dir.")
    parser.add_argument("--output_file", type=str, default="out.json", help="Output JSON file.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of model indices to run (e.g., '0,2,3')")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--save_after_each_model", action="store_true", help="Save results after each model is processed")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random images to sample for evaluation (default: 5). Set to 0 for all images.")
    args = parser.parse_args()
    
    if args.cpu_only:
        logger.info("Forcing CPU usage as requested by --cpu_only flag.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    
    try:
        with open("models.json", 'r') as f: model_configs_all = json.load(f)
        logger.info(f"Loaded {len(model_configs_all)} model configurations from models.json")
    except FileNotFoundError: logger.error("models.json not found."); return
    except Exception as e: logger.error(f"Error loading models.json: {e}"); return
    
    model_configs_to_run = model_configs_all
    if args.models:
        try:
            selected_indices = [int(idx.strip()) for idx in args.models.split(",")]
            model_configs_to_run = [model_configs_all[i] for i in selected_indices if 0 <= i < len(model_configs_all)]
            logger.info(f"Filtered to {len(model_configs_to_run)} models based on indices: {selected_indices}")
            if len(model_configs_to_run) != len(selected_indices): logger.warning("Some model indices out of range.")
        except Exception as e: logger.error(f"Error parsing --models '{args.models}': {e}"); return

    # --- Initialize ConceptualCaptionsDataset ---
    data_cache_dir = Path(args.test_dir) # test_dir is now cache_dir
    logger.info(f"Initializing ConceptualCaptionsDataset with cache_dir: {data_cache_dir}")
    try:
        from datasets import load_dataset 
        dataset = load_dataset("google-research-datasets/conceptual_captions", split="validation")
        downloaded_indices = sorted([int(p.stem) for p in data_cache_dir.glob("*.jpg") if p.stem.isdigit()])
        num_samples = 5
        downloaded_indices = downloaded_indices[:num_samples]
        transform = A.Compose([
            A.LongestMaxSize(max_size=512),
            A.HorizontalFlip(p=0.5),
        ])

        conceptual_dataset = ConceptualCaptionsDataset(
            dataset,
            downloaded_indices,
            cache_dir=data_cache_dir,
            transform=transform
        )
        if len(conceptual_dataset) == 0:
            logger.error(f"ConceptualCaptionsDataset is empty. Check cache_dir '{data_cache_dir}' and dataset logic.")
            return
        logger.info(f"Initialized ConceptualCaptionsDataset with {len(conceptual_dataset)} effectively available items.")
    except Exception as e:
        logger.error(f"Error initializing ConceptualCaptionsDataset: {e}"); traceback.print_exc(); return

    dataset_size = len(conceptual_dataset)
    indices_to_process_from_custom_dataset = list(range(dataset_size))

    if args.num_samples > 0 and dataset_size > args.num_samples:
        logger.info(f"Randomly sampling {args.num_samples} items from ConceptualCaptionsDataset (size: {dataset_size}).")
        sampled_custom_dataset_indices = random.sample(indices_to_process_from_custom_dataset, args.num_samples)
    elif args.num_samples == 0:
        sampled_custom_dataset_indices = indices_to_process_from_custom_dataset
    else: # num_samples < 0 or num_samples > dataset_size
        sampled_custom_dataset_indices = indices_to_process_from_custom_dataset
    
    logger.info(f"Number of dataset items to process: {len(sampled_custom_dataset_indices)}")
    if not sampled_custom_dataset_indices: logger.warning("No samples selected to process."); return

    results = {}
    if os.path.exists(args.output_file) and args.save_after_each_model:
        try:
            with open(args.output_file, 'r') as f: results = json.load(f)
            logger.info(f"Loaded existing results from '{args.output_file}'.")
        except Exception as e:
            logger.warning(f"Could not load existing results from '{args.output_file}', starting fresh: {e}"); results = {}
    
    for idx_config, config in enumerate(model_configs_to_run):
        model_identifier = config.get("adapter_hub_id", config.get("decoder_name", f"unknown_model_{idx_config}")).replace("/", "_")
        logger.info(f"\nProcessing model {idx_config+1}/{len(model_configs_to_run)}: {model_identifier}")
        
        current_model_results = results.get(model_identifier, [])
        # More robust resume: track by a unique image identifier from the dataset item
        processed_item_identifiers = {entry["image_path"] for entry in current_model_results if "image_path" in entry}

        model_obj, processor, tokenizer = None, None, None
        try:
            model_obj, processor, tokenizer = load_model_for_inference(config)
            
            items_to_generate_for_this_run = []
            for dataset_idx in sampled_custom_dataset_indices:
                try:
                    item = conceptual_dataset[dataset_idx]
                    log_image_path = item.get('log_image_path', f"item_dsidx_{dataset_idx}") # Get identifier
                    if log_image_path not in processed_item_identifiers:
                        items_to_generate_for_this_run.append(item)
                    else:
                        logger.debug(f"Skipping already processed item: {log_image_path}")
                except IndexError: logger.warning(f"Dataset index {dataset_idx} out of bounds.")
                except KeyError as e_key: logger.warning(f"KeyError {e_key} for dataset index {dataset_idx}.")


            if not items_to_generate_for_this_run and args.save_after_each_model and model_identifier in results:
                logger.info(f"All selected samples for {model_identifier} already processed in loaded results.")
                continue # Skip to next model if all done

            logger.info(f"Generating for {len(items_to_generate_for_this_run)} items with {model_identifier}")
            for item in tqdm(items_to_generate_for_this_run, desc=f"Generating with {model_identifier}"):
                try:
                    pil_image = item['image']         
                    base_caption = item['text']   
                    log_image_path = item['image_path'] 
                    
                    # Determine device for inputs from the model itself
                    input_device = next(model_obj.parameters()).device

                    caption = generate_caption(model_obj, processor, tokenizer, pil_image, log_image_path)
                    
                    current_model_results.append({
                        "image_path": log_image_path,
                        "base_caption": base_caption,
                        "generated_caption": caption
                    })
                    
                    if len(current_model_results) % 20 == 0 and args.save_after_each_model: # Save every 20 new captions
                        results[model_identifier] = current_model_results # Update main results
                        with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
                        logger.info(f"Saved intermediate results to '{args.output_file}'")
                
                except KeyError as e_key:
                    logger.error(f"KeyError accessing item from ConceptualCaptionsDataset: {e_key}. Item keys: {item.keys() if isinstance(item, dict) else 'N/A'}. Skipping item.")
                    # Find a way to log error if item structure is unexpected
                    log_path_err = item.get('log_image_path', f"unknown_item_error") if isinstance(item, dict) else f"unknown_item_error"
                    current_model_results.append({ "image_path": log_path_err, "error": f"KeyError: {e_key}"})

                except Exception as e_img:
                    logger.error(f"Error processing item (log_path: {item.get('log_image_path', 'N/A') if isinstance(item, dict) else 'N/A'}) with {model_identifier}: {e_img}")
                    log_path_err = item.get('log_image_path', f"unknown_item_error") if isinstance(item, dict) else f"unknown_item_error"
                    current_model_results.append({
                        "image_path": log_path_err,
                        "base_caption": item.get('caption_text', "") if isinstance(item, dict) else "",
                        "generated_caption": f"ERROR: {str(e_img)[:100]}..."
                    })
            
            results[model_identifier] = current_model_results # Final update for this model
            if args.save_after_each_model:
                with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
                logger.info(f"Saved results after processing {model_identifier} to '{args.output_file}'")
        
        except Exception as e_model_load:
            logger.error(f"FATAL: Error loading or processing with model {model_identifier}: {e_model_load}")
            traceback.print_exc()
            if model_identifier not in results: 
                 results[model_identifier] = [{"error": f"Model loading failed: {str(e_model_load)[:200]}..."}]
            if args.save_after_each_model:
                with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)

        finally: 
            if model_obj is not None: del model_obj
            if 'processor' in locals() and processor is not None: del processor
            if 'tokenizer' in locals() and tokenizer is not None: del tokenizer
            if torch.cuda.is_available():
                logger.debug("Final CUDA cache clear for model.")
                torch.cuda.empty_cache(); gc.collect(); time.sleep(1) 
            
    try:
        with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
        logger.info(f"All processing complete. Final results saved to '{args.output_file}'")
    except Exception as e_final_save:
        logger.error(f"Error during final save to '{args.output_file}': {e_final_save}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()
    main()


