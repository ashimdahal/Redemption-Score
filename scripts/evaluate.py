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
        logging.FileHandler("inference_log.txt", mode='w') # Overwrite log file each run
    ]
)
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader, Subset # Added DataLoader and Subset

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
    BertLMHeadModel,
    GPT2LMHeadModel,
    MllamaProcessor, 
    BertModel
)

from peft import PeftModel

import albumentations as A

from multimodel_utils import MultimodalModel, MultimodalCollator 
from data import ConceptualCaptionsDataset # Your custom dataset

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
        
        # Modified section: Always wrap with MultimodalModel for consistency
        underlying_model_for_check = model_to_return.base_model.model if isinstance(model_to_return, PeftModel) else model_to_return

        # Always use MultimodalModel for Qwen2VL and other models for consistent handling
        logger.info(f"Wrapping {type(underlying_model_for_check).__name__} in MultimodalModel for consistent handling.")
        return MultimodalModel(processor=processor, decoder=model_to_return, tokenizer=tokenizer), processor, tokenizer
    
    except Exception as e:
        logger.error(f"Error in load_model_for_inference for config '{config.get('decoder_name', 'N/A')}': {e}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate captions using trained models")
    parser.add_argument("--test_dir", type=str, default="./valid_dataset/", help="Directory for ConceptualCaptionsDataset cache_dir.")
    parser.add_argument("--output_file", type=str, default="out.json", help="Output JSON file.")
    parser.add_argument("--models_config", type=str, default="models.json", help="Path to models configuration JSON file.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of model indices to run (e.g., '0,2,3')")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--save_after_each_model", action="store_true", help="Save results after each model is processed")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random images to sample for evaluation (default: 5). Set to 0 for all images.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for DataLoader (default: 5).")
    args = parser.parse_args()

    global GLOBAL_DEVICE_PRIMARY 
    if args.cpu_only:
        logger.info("Forcing CPU usage as requested by --cpu_only flag.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        GLOBAL_DEVICE_PRIMARY = torch.device("cpu") 
        if torch.cuda.is_available(): 
             logger.warning("CUDA is still reported as available after setting CUDA_VISIBLE_DEVICES=''. Ensure PyTorch respects this.")

    try:
        with open(args.models_config, 'r') as f: model_configs_all = json.load(f)
        logger.info(f"Loaded {len(model_configs_all)} model configurations from {args.models_config}")
    except FileNotFoundError: logger.error(f"{args.models_config} not found."); return
    except Exception as e: logger.error(f"Error loading {args.models_config}: {e}"); return

    model_configs_to_run = model_configs_all
    if args.models:
        try:
            selected_indices = [int(idx.strip()) for idx in args.models.split(",")]
            model_configs_to_run = [model_configs_all[i] for i in selected_indices if 0 <= i < len(model_configs_all)]
            logger.info(f"Filtered to {len(model_configs_to_run)} models based on indices: {selected_indices}")
            if len(model_configs_to_run) != len(selected_indices): logger.warning("Some model indices were out of range and skipped.")
        except Exception as e: logger.error(f"Error parsing --models '{args.models}': {e}"); return

    data_cache_dir = Path(args.test_dir)
    logger.info(f"Initializing ConceptualCaptionsDataset with cache_dir: {data_cache_dir}")
    try:
        from datasets import load_dataset 
        dataset_hf = load_dataset("google-research-datasets/conceptual_captions", split="validation", trust_remote_code=True)
        
        downloaded_indices_all = sorted([int(p.stem) for p in data_cache_dir.glob("*.jpg") if p.stem.isdigit()])
        if not downloaded_indices_all:
            logger.warning(f"No images like '*.jpg' with numeric names found in {data_cache_dir}. Dataset might be empty or use different naming.")

        transform = A.Compose([ A.LongestMaxSize(max_size=512), A.HorizontalFlip(p=0.5) ])
        conceptual_dataset = ConceptualCaptionsDataset( dataset_hf, downloaded_indices_all, cache_dir=data_cache_dir, transform=transform )
        if len(conceptual_dataset) == 0:
            logger.error(f"ConceptualCaptionsDataset is empty. Check cache_dir '{data_cache_dir}' and dataset logic.")
            return
        logger.info(f"Initialized ConceptualCaptionsDataset with {len(conceptual_dataset)} effectively available items.")
    except Exception as e:
        logger.error(f"Error initializing ConceptualCaptionsDataset: {e}"); traceback.print_exc(); return

    dataset_size = len(conceptual_dataset)
    indices_to_process_from_custom_dataset = list(range(dataset_size)) 

    if args.num_samples > 0 and args.num_samples < dataset_size :
        logger.info(f"Randomly sampling {args.num_samples} items from ConceptualCaptionsDataset (size: {dataset_size}).")
        sampled_custom_dataset_indices = random.sample(indices_to_process_from_custom_dataset, args.num_samples)
    elif args.num_samples == 0: 
        logger.info(f"Processing all {dataset_size} items from ConceptualCaptionsDataset (--num_samples is 0).")
        sampled_custom_dataset_indices = indices_to_process_from_custom_dataset
    else: 
        logger.info(f"Processing all {dataset_size} items from ConceptualCaptionsDataset (num_samples: {args.num_samples} implies all).")
        sampled_custom_dataset_indices = indices_to_process_from_custom_dataset
    
    logger.info(f"Total number of dataset indices selected for potential processing: {len(sampled_custom_dataset_indices)}")
    if not sampled_custom_dataset_indices: logger.warning("No samples selected to process overall."); return

    results = {}
    if os.path.exists(args.output_file) and args.save_after_each_model:
        try:
            with open(args.output_file, 'r') as f: results = json.load(f)
            logger.info(f"Loaded existing results from '{args.output_file}'.")
        except Exception as e:
            logger.warning(f"Could not load existing results from '{args.output_file}', starting fresh: {e}"); results = {}
    
    for idx_config, config in enumerate(model_configs_to_run):
        model_identifier = config.get("model_identifier_for_results", 
                                      config.get("adapter_hub_id", config.get("decoder_name", f"unknown_model_{idx_config}")).replace("/", "_"))
        logger.info(f"\nProcessing model {idx_config+1}/{len(model_configs_to_run)}: {model_identifier}")
        
        current_model_results = results.get(model_identifier, [])
        processed_item_identifiers = {entry["image_path"] for entry in current_model_results if "image_path" in entry}
        num_items_processed_before_this_model_run = len(processed_item_identifiers)

        model_obj = None 
        
        try:
            actual_indices_for_dataloader = []
            for dataset_idx in sampled_custom_dataset_indices: 
                try:
                    item_for_check = conceptual_dataset[dataset_idx] 
                    log_image_path_for_check = item_for_check.get('image_path', f"item_dsidx_{dataset_idx}")
                    if log_image_path_for_check not in processed_item_identifiers:
                        actual_indices_for_dataloader.append(dataset_idx)
                except IndexError: 
                    logger.warning(f"Dataset index {dataset_idx} out of bounds during pre-filtering for DataLoader for model {model_identifier}.")
                except Exception as e_filter:
                    logger.error(f"Error getting item {dataset_idx} for pre-filtering: {e_filter}")

            if not actual_indices_for_dataloader:
                logger.info(f"All selected samples for {model_identifier} already processed and found in loaded results. Skipping model.")
                continue
            
            logger.info(f"Number of new items to process for {model_identifier}: {len(actual_indices_for_dataloader)}")

            subset_for_model = Subset(conceptual_dataset, actual_indices_for_dataloader)
            model_obj, processor, tokenizer = load_model_for_inference(config) 

            collator = MultimodalCollator(processor, tokenizer)
            dataloader = DataLoader(
                subset_for_model, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=0,
                collate_fn=collator
            ) 
            
            num_newly_processed_in_session = 0
            for batch_data in tqdm(dataloader, desc=f"Generating with {model_identifier} (Batch Size: {args.batch_size})"):
                try:
                    batch_results = model_obj.generate_caption(batch_data)
                    print(batch_results)

                    if batch_results and isinstance(batch_results, list):
                        current_model_results.extend(batch_results)
                        num_newly_processed_in_session += len(batch_results)
                    else:
                        logger.error(f"model.generate_caption did not return a list of results for the batch. Received: {type(batch_results)}. Skipping batch results.")
                        for item_in_failed_batch in batch_data: # batch_data is List[Dict]
                            current_model_results.append({
                                "image_path": item_in_failed_batch.get("image_path", "unknown_path_in_failed_batch"),
                                "base_caption": item_in_failed_batch.get("text", ""),
                                "generated_caption": "ERROR: Batch processing failed or returned invalid format."
                            })
                    
                    if args.save_after_each_model and \
                       (len(current_model_results) - num_items_processed_before_this_model_run) > 0 and \
                       ((len(current_model_results) - num_items_processed_before_this_model_run) % 20 < args.batch_size) and \
                       (len(current_model_results) - num_items_processed_before_this_model_run) // 20 > (num_newly_processed_in_session - len(batch_results if batch_results and isinstance(batch_results, list) else [])) // 20 :
                        results[model_identifier] = current_model_results
                        with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
                        logger.info(f"Saved intermediate results ({len(current_model_results)} total for model) to '{args.output_file}'")
                
                except Exception as e_batch_call: 
                    logger.error(f"Error calling model.generate_caption for a batch with {model_identifier}: {e_batch_call}")
                    traceback.print_exc()
                    for item_in_failed_batch in batch_data: # batch_data is List[Dict]
                         current_model_results.append({
                            "image_path": item_in_failed_batch.get("image_path", "unknown_path_in_failed_batch_exception"),
                            "base_caption": item_in_failed_batch.get("text", ""),
                            "generated_caption": f"ERROR_CALLING_BATCH_GENERATE: {str(e_batch_call)[:100]}..."
                        })

            results[model_identifier] = current_model_results 
            if args.save_after_each_model:
                with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
                logger.info(f"Saved results after processing all batches for {model_identifier} to '{args.output_file}'")
        
        except Exception as e_model_load_or_process:
            logger.error(f"FATAL: Error during loading or processing with model {model_identifier}: {e_model_load_or_process}")
            traceback.print_exc()
            if model_identifier not in results: 
                results[model_identifier] = [{"error": f"Model failed: {str(e_model_load_or_process)[:200]}..."}]
            if args.save_after_each_model: 
                with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
        finally:
            if model_obj is not None:
                logger.debug(f"Deleting model object for {model_identifier}")
                del model_obj 
            
            if torch.cuda.is_available():
                logger.debug(f"Clearing CUDA cache after model {model_identifier}.")
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
