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
    GitForCausalLM, # Important for the fix
    ViTImageProcessor,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    # BitsAndBytesConfig is not used in the user's new load_model_for_inference
    # Assuming these are imported if used by vision_encoder_decoder_compatible
    BertLMHeadModel, 
    GPT2LMHeadModel, 
    MllamaProcessor,
    BertModel      
)

from peft import PeftModel, PeftConfig # PeftConfig is imported as per user's code

import albumentations as A # Imported as per user's code

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
    # LlamaForCausalLM was in previous user versions, but not in the latest one.
    # Sticking to the latest provided list.
    MllamaForConditionalGeneration,
    Blip2ForConditionalGeneration, 
    Qwen2_5_VLForConditionalGeneration 
)

GLOBAL_DEVICE_PRIMARY = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_for_inference(config):
    """
    Load a pretrained model for inference using the adapter hub ID.
    Generalized solution to handle any model class that doesn't support device_map='auto'.
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
    base_model_obj = None
    manually_moved_to_device = False  # Track if we manually moved model to device

    try:
        # Load the processor
        processor_name = config["processor_name"]
        logger.info(f"Loading processor: {processor_name}")
        if "processor_class" in config and config["processor_class"]:
            ProcessorClass = eval(config["processor_class"])
            processor = ProcessorClass.from_pretrained(processor_name, trust_remote_code=True)
        else:
            processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
        
        if "Qwen" in processor_name:
            logger.info(f"Qwen processor detected. Re-loading with max_pixels.")
            processor = AutoProcessor.from_pretrained(config['processor_name'], max_pixels=512*28*28, trust_remote_code=True)
        
        # Determine device and dtype for base model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_for_base_model = torch.float32  # Default for CPU

        decoder_name = config["decoder_name"]
        DecoderClass = eval(config["decoder_class"])
        
        load_kwargs_base = {"trust_remote_code": True}

        if device == "cuda":
            # First, try to load with device_map='auto'
            if device == "cuda":
                logger.info(f"Attempting to load model with device_map='auto' and bfloat16")
                load_kwargs_with_device_map = {**load_kwargs_base, "device_map": "auto"}
                
                try:
                    # Try loading with device_map='auto' and bfloat16
                    base_model_obj = DecoderClass.from_pretrained(
                        decoder_name, 
                        torch_dtype=torch.bfloat16, 
                        **load_kwargs_with_device_map
                    )
                    dtype_for_base_model = torch.bfloat16
                    logger.info(f"Successfully loaded model with device_map='auto' and bfloat16")
                
                except (ValueError, RuntimeError) as e:
                    # Check if error is specifically about device_map='auto'
                    if "_no_split_modules" in str(e) or "device_map='auto'" in str(e):
                        logger.info(f"Model doesn't support device_map='auto'. Error: {e}")
                        logger.info(f"Falling back to load model without device_map and then move to device")
                        
                        # Try with bfloat16 first, without device_map
                        try:
                            base_model_obj = DecoderClass.from_pretrained(
                                decoder_name, 
                                torch_dtype=torch.bfloat16,
                                **load_kwargs_base  # Without device_map
                            )
                            dtype_for_base_model = torch.bfloat16
                            
                            # Explicitly move to cuda
                            base_model_obj = base_model_obj.to(device)
                            manually_moved_to_device = True
                            logger.info(f"Successfully loaded model without device_map and moved to {device} with bfloat16")
                            
                        except Exception as e_bf16:
                            logger.warning(f"bfloat16 without device_map failed: {e_bf16}. Trying float16.")
                            
                            # Try with float16, without device_map
                            base_model_obj = DecoderClass.from_pretrained(
                                decoder_name, 
                                torch_dtype=torch.float16,
                                **load_kwargs_base  # Without device_map
                            )
                            dtype_for_base_model = torch.float16
                            
                            # Explicitly move to cuda
                            base_model_obj = base_model_obj.to(device)
                            manually_moved_to_device = True
                            logger.info(f"Successfully loaded model without device_map and moved to {device} with float16")
                    
                    else:  # Error is not related to device_map='auto'
                        logger.warning(f"bfloat16 with device_map='auto' failed, but not due to device_map compatibility: {e}")
                        logger.info(f"Trying with float16 and device_map='auto'")
                        
                        # Try with float16 and device_map='auto'
                        base_model_obj = DecoderClass.from_pretrained(
                            decoder_name, 
                            torch_dtype=torch.float16, 
                            **load_kwargs_with_device_map
                        )
                        dtype_for_base_model = torch.float16
                        logger.info(f"Successfully loaded model with device_map='auto' and float16")
        
        else:  # CPU
            logger.info(f"Target device: CPU. Loading base model with torch_dtype=torch.float32")
            base_model_obj = DecoderClass.from_pretrained(
                decoder_name, 
                torch_dtype=torch.float32, 
                **load_kwargs_base
            )
            dtype_for_base_model = torch.float32
        
        device_info = base_model_obj.hf_device_map if hasattr(base_model_obj, 'hf_device_map') else base_model_obj.device
        logger.info(f"Base model loaded with dtype: {dtype_for_base_model}. Initial device distribution: {device_info}")

        # Load the adapter if adapter_hub_id is provided
        model_to_return = base_model_obj
        if config.get("adapter_hub_id"):
            adapter_path_or_id = config["adapter_hub_id"]
            if os.path.exists(adapter_path_or_id):
                logger.info(f"Loading PEFT adapter from local path: {adapter_path_or_id}")
            else:
                logger.info(f"Loading PEFT adapter from Hugging Face Hub ID: {adapter_path_or_id}")

            try:
                model_to_return = PeftModel.from_pretrained(base_model_obj, adapter_path_or_id)
                logger.info(f"PEFT adapter '{adapter_path_or_id}' loaded and applied.")
            except Exception as e_peft:
                logger.error(f"Failed to load PEFT adapter '{adapter_path_or_id}': {e_peft}")
                logger.warning("Proceeding with base model only due to PEFT loading error.")
        else:
            logger.info("No 'adapter_hub_id' found in config. Using base model directly.")

        # Only move to device if we haven't already done so manually and it makes sense to do so
        if not manually_moved_to_device:
            try:
                current_final_device = next(model_to_return.parameters()).device
                if str(current_final_device) != device and device != "cpu":
                     if not hasattr(base_model_obj, 'hf_device_map'):  # Only move if not sharded
                        logger.info(f"Moving final model to specified device: {device}")
                        model_to_return = model_to_return.to(device)
                elif str(current_final_device) != device and device == "cpu":
                    logger.info(f"Moving final model to specified device: {device}")
                    model_to_return = model_to_return.to(device)
            except StopIteration:
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
        
        # Handle different model types for potential wrapping
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
        traceback.print_exc()
        raise

# --- User's provided caption generation function ---
def generate_caption(model, processor, tokenizer, pil_image: Image.Image, image_identifier: str, max_length=50, num_beams=5):
    """
    Generate caption for a single PIL image.
    image_identifier is used for logging purposes.
    """
    try:
        model_device = next(model.parameters()).device
        logger.debug(f"Generating caption for '{image_identifier}' on device: {model_device}")
        
        model.eval() 
        
        if hasattr(model, 'generate_caption') and callable(model.generate_caption):
            logger.debug(f"Using model.generate_caption method for '{image_identifier}'.")
            caption = model.generate_caption(pil_image, max_length=max_length, num_beams=num_beams)
            return caption.strip() if isinstance(caption, str) else caption
        else: 
            logger.debug(f"Using generic model.generate method for '{image_identifier}'.")
            with torch.cuda.amp.autocast(enabled=(model_device.type == "cuda")):  
                inputs = processor(images=pil_image, return_tensors="pt").to(model_device)
                
                with torch.no_grad():
                    if hasattr(model, 'generate'):
                        try:
                            output_ids = model.generate(
                                **inputs, max_length=max_length, num_beams=num_beams, early_stopping=True
                            )
                        except RuntimeError as e_gen: 
                            if "CUDA out of memory" in str(e_gen):
                                logger.warning(f"CUDA OOM for '{image_identifier}', retrying with reduced beams.")
                                if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
                                output_ids = model.generate(
                                    **inputs, max_length=max_length, num_beams=max(1, num_beams // 2), early_stopping=True
                                )
                            else: raise 
                        
                        if hasattr(tokenizer, 'batch_decode'):
                            caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                        elif hasattr(processor, 'decode'): 
                            caption = processor.decode(output_ids[0], skip_special_tokens=True)
                        else:
                            logger.error("Neither tokenizer.batch_decode nor processor.decode found.")
                            return "Decoding error"
                        return caption.strip()
                    else:
                        logger.error(f"Model type {type(model)} does not have a 'generate' method.")
                        return "Model doesn't support .generate"
        
    except RuntimeError as e_runtime: 
        if "CUDA out of memory" in str(e_runtime) and model_device.type == "cuda": # Check model_device was cuda
            logger.warning(f"CUDA OOM for '{image_identifier}', attempting fallback to CPU for this image.")
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
            cpu_device = torch.device("cpu")
            try:
                original_device = next(model.parameters()).device # Store original device
                model.to(cpu_device) 
                logger.info(f"Moved model to CPU for '{image_identifier}' due to OOM.")
                inputs = processor(images=pil_image, return_tensors="pt").to(cpu_device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_length=max_length, num_beams=max(1, num_beams // 2), early_stopping=True
                    )
                logger.info(f"Attempting to move model back to original device: {original_device}")
                model.to(original_device)

                if hasattr(tokenizer, 'batch_decode'):
                    caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                else:
                    caption = processor.decode(output_ids[0], skip_special_tokens=True)
                return caption.strip()
            except Exception as e_cpu_fallback:
                logger.error(f"Error during CPU fallback for '{image_identifier}': {e_cpu_fallback}")
                traceback.print_exc()
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

    # --- Initialize ConceptualCaptionsDataset (User's specific initialization) ---
    data_cache_dir = Path(args.test_dir) 
    logger.info(f"Initializing ConceptualCaptionsDataset with cache_dir: {data_cache_dir}")
    try:
        from datasets import load_dataset 
        dataset_hf = load_dataset("google-research-datasets/conceptual_captions", split="validation", trust_remote_code=True) # User specified "validation"
        
        downloaded_indices_all = sorted([int(p.stem) for p in data_cache_dir.glob("*.jpg") if p.stem.isdigit()])
        if not downloaded_indices_all:
             logger.warning(f"No images like '*.jpg' with numeric names found in {data_cache_dir}. Dataset might be empty or use different naming.")
        
        transform = A.Compose([
            A.LongestMaxSize(max_size=512), # User's specified transform
            A.HorizontalFlip(p=0.5),
        ])

        conceptual_dataset = ConceptualCaptionsDataset(
            dataset_hf, 
            downloaded_indices_all, 
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
        logger.info(f"Processing all {dataset_size} items from ConceptualCaptionsDataset (--num_samples is 0).")
        sampled_custom_dataset_indices = indices_to_process_from_custom_dataset
    else: 
        logger.info(f"Processing all {dataset_size} items from ConceptualCaptionsDataset (num_samples: {args.num_samples}).")
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
        processed_item_identifiers = {entry["image_path"] for entry in current_model_results if "image_path" in entry}

        model_obj, processor_obj, tokenizer_obj = None, None, None 
        try:
            model_obj, processor_obj, tokenizer_obj = load_model_for_inference(config)
            
            items_to_generate_for_this_run = []
            for dataset_idx in sampled_custom_dataset_indices:
                try:
                    item = conceptual_dataset[dataset_idx]
                    log_image_path = item.get('image_path', f"item_dsidx_{dataset_idx}") 
                    if log_image_path not in processed_item_identifiers:
                        items_to_generate_for_this_run.append(item)
                    else:
                        logger.debug(f"Skipping already processed item: {log_image_path}")
                except IndexError: logger.warning(f"Dataset index {dataset_idx} out of bounds.")
                except KeyError as e_key: logger.warning(f"KeyError {e_key} for dataset index {dataset_idx} when preparing items.")


            if not items_to_generate_for_this_run and args.save_after_each_model and model_identifier in results and len(current_model_results) >= len(sampled_custom_dataset_indices):
                logger.info(f"All selected samples for {model_identifier} already processed in loaded results.")
                continue 

            logger.info(f"Generating for {len(items_to_generate_for_this_run)} items with {model_identifier}")
            for item in tqdm(items_to_generate_for_this_run, desc=f"Generating with {model_identifier}"):
                try:
                    pil_image = item['image']         
                    base_caption = item['text']   
                    log_image_path = item['image_path'] 
                    
                    input_device_for_gen = next(model_obj.parameters()).device

                    caption = generate_caption(model_obj, processor_obj, tokenizer_obj, pil_image, log_image_path) 
                    
                    current_model_results.append({
                        "image_path": log_image_path,
                        "base_caption": base_caption,
                        "generated_caption": caption
                    })
                    
                    if (len(current_model_results) - len(processed_item_identifiers) ) % 20 == 0 and args.save_after_each_model: 
                        results[model_identifier] = current_model_results 
                        with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
                        logger.info(f"Saved intermediate results to '{args.output_file}'")
                
                except KeyError as e_key:
                    logger.error(f"KeyError accessing item from ConceptualCaptionsDataset: {e_key}. Item keys: {item.keys() if isinstance(item, dict) else 'N/A'}. Skipping item.")
                    log_path_err = item.get('image_path', f"unknown_item_error_key_{e_key}") if isinstance(item, dict) else f"unknown_item_error_key_{e_key}"
                    current_model_results.append({ "image_path": log_path_err, "error": f"KeyError: {e_key}"})

                except Exception as e_img:
                    logger.error(f"Error processing item (log_path: {item.get('image_path', 'N/A') if isinstance(item, dict) else 'N/A'}) with {model_identifier}: {e_img}")
                    traceback.print_exc()
                    log_path_err = item.get('image_path', f"unknown_item_error_exc") if isinstance(item, dict) else f"unknown_item_error_exc"
                    current_model_results.append({
                        "image_path": log_path_err,
                        "base_caption": item.get('text', "") if isinstance(item, dict) else "",
                        "generated_caption": f"ERROR: {str(e_img)[:100]}..."
                    })
            
            results[model_identifier] = current_model_results 
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
            if 'model_obj' in locals() and model_obj is not None: del model_obj # Check if defined before deleting
            if 'processor_obj' in locals() and processor_obj is not None: del processor_obj 
            if 'tokenizer_obj' in locals() and tokenizer_obj is not None: del tokenizer_obj 
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


