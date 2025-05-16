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
    manually_moved_to_device = False 

    try:
        processor_name = config["processor_name"]
        logger.info(f"Loading processor: {processor_name}")
        if "processor_class" in config and config["processor_class"]:
            ProcessorClass = eval(config["processor_class"])
            processor = ProcessorClass.from_pretrained(processor_name, trust_remote_code=True)
        else:
            processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
        
        if "Qwen" in processor_name: # Specific handling for Qwen processor
            logger.info(f"Qwen processor detected. Re-loading with increased max_pixels.")
            # Note: max_pixels might need adjustment based on typical image sizes and model capabilities
            processor = AutoProcessor.from_pretrained(config['processor_name'], max_pixels=int(512*28*28*1.5), trust_remote_code=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_for_base_model = torch.float32  

        decoder_name = config["decoder_name"]
        DecoderClass = eval(config["decoder_class"])
        
        load_kwargs_base = {"trust_remote_code": True}

        if device == "cuda":
            logger.info(f"Attempting to load model with device_map='auto' and bfloat16")
            load_kwargs_with_device_map = {**load_kwargs_base, "device_map": "auto"}
            
            try:
                base_model_obj = DecoderClass.from_pretrained(
                    decoder_name, 
                    torch_dtype=torch.bfloat16, 
                    **load_kwargs_with_device_map
                )
                dtype_for_base_model = torch.bfloat16
                logger.info(f"Successfully loaded model with device_map='auto' and bfloat16")
            
            except (ValueError, RuntimeError) as e:
                if "_no_split_modules" in str(e) or "device_map='auto'" in str(e):
                    logger.info(f"Model doesn't support device_map='auto'. Error: {e}")
                    logger.info(f"Falling back to load model without device_map and then move to device")
                    
                    try: # Try bfloat16 without device_map
                        base_model_obj = DecoderClass.from_pretrained(
                            decoder_name, 
                            torch_dtype=torch.bfloat16,
                            **load_kwargs_base
                        )
                        dtype_for_base_model = torch.bfloat16
                        base_model_obj = base_model_obj.to(device)
                        manually_moved_to_device = True
                        logger.info(f"Successfully loaded model without device_map and moved to {device} with bfloat16")
                        
                    except Exception as e_bf16: # Fallback to float16 without device_map
                        logger.warning(f"bfloat16 without device_map failed: {e_bf16}. Trying float16.")
                        base_model_obj = DecoderClass.from_pretrained(
                            decoder_name, 
                            torch_dtype=torch.float16,
                            **load_kwargs_base
                        )
                        dtype_for_base_model = torch.float16
                        base_model_obj = base_model_obj.to(device)
                        manually_moved_to_device = True
                        logger.info(f"Successfully loaded model without device_map and moved to {device} with float16")
                
                else: # bfloat16 with device_map='auto' failed for other reasons, try float16 with device_map='auto'
                    logger.warning(f"bfloat16 with device_map='auto' failed (not due to device_map compatibility): {e}")
                    logger.info(f"Trying with float16 and device_map='auto'")
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

        model_to_return = base_model_obj
        if config.get("adapter_hub_id"):
            adapter_path_or_id = config["adapter_hub_id"]
            logger.info(f"Attempting to load PEFT adapter from: {adapter_path_or_id} (local if exists, else Hub)")
            try:
                model_to_return = PeftModel.from_pretrained(base_model_obj, adapter_path_or_id)
                logger.info(f"PEFT adapter '{adapter_path_or_id}' loaded and applied.")
            except Exception as e_peft:
                logger.error(f"Failed to load PEFT adapter '{adapter_path_or_id}': {e_peft}")
                logger.warning("Proceeding with base model only due to PEFT loading error.")
        else:
            logger.info("No 'adapter_hub_id' found in config. Using base model directly.")

        if not manually_moved_to_device and device != "cpu": # Only move if not already manually moved and target is not CPU (already handled for CPU)
            try:
                current_final_device = next(iter(model_to_return.parameters())).device # Use iter for safety
                if str(current_final_device) != device:
                    if not hasattr(base_model_obj, 'hf_device_map'): # Only move if not sharded by device_map
                        logger.info(f"Moving final model from {current_final_device} to specified device: {device}")
                        model_to_return = model_to_return.to(device)
                    else:
                        logger.info(f"Model appears to be sharded (hf_device_map exists). Final device distribution: {model_to_return.hf_device_map}")
            except StopIteration:
                logger.warning("Model has no parameters. Cannot determine or move device.")
        
        model_to_return.eval() 
        
        final_device_check = "unknown"
        try: final_device_check = next(iter(model_to_return.parameters())).device
        except: pass
        logger.info(f"Final model ready. Effective device: {final_device_check}")

        tokenizer_name = config.get("tokenizer_name", processor_name)
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set tokenizer pad_token to eos_token: {tokenizer.pad_token}")
            else:
                new_pad_token = '[PAD]'
                tokenizer.add_special_tokens({'pad_token': new_pad_token})
                # If the model is an AutoModelForCausalLM, its embedding layer might need resizing
                if hasattr(model_to_return, 'resize_token_embeddings'):
                    model_to_return.resize_token_embeddings(len(tokenizer))
                logger.info(f"Added new pad_token: {new_pad_token} and resized embeddings if applicable.")
        
        underlying_model_for_check = model_to_return.base_model.model if isinstance(model_to_return, PeftModel) else model_to_return
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
    parser.add_argument("--num_samples", type=int, default=0, help="Number of random images to sample for evaluation (0 = all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    args = parser.parse_args()

    global GLOBAL_DEVICE_PRIMARY
    if args.cpu_only:
        logger.info("Forcing CPU usage via --cpu_only")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        GLOBAL_DEVICE_PRIMARY = torch.device("cpu")

    # ------------------------------------------------------------------
    # 1. Load model configurations
    # ------------------------------------------------------------------
    try:
        with open(args.models_config) as f:
            model_configs_all = json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", args.models_config, e)
        return
    logger.info("Loaded %d model configs", len(model_configs_all))

    if args.models:
        indices = [int(x) for x in args.models.split(",")]
        model_configs_to_run = [model_configs_all[i] for i in indices if 0 <= i < len(model_configs_all)]
    else:
        model_configs_to_run = model_configs_all

    # ------------------------------------------------------------------
    # 2. Build Conceptual Captions dataset (cached locally)
    # ------------------------------------------------------------------
    data_cache_dir = Path(args.test_dir)
    from datasets import load_dataset
    dataset_hf = load_dataset("google-research-datasets/conceptual_captions", split="validation")

    downloaded_indices_all = sorted([int(p.stem) for p in data_cache_dir.glob("*.jpg") if p.stem.isdigit()])
    transform = A.Compose([A.LongestMaxSize(max_size=512), A.HorizontalFlip(p=0.5)])
    conceptual_dataset = ConceptualCaptionsDataset(dataset_hf, downloaded_indices_all, cache_dir=data_cache_dir, transform=transform)

    dataset_size = len(conceptual_dataset)
    if args.num_samples == 0 or args.num_samples >= dataset_size:
        sampled_indices = list(range(dataset_size))
    else:
        sampled_indices = random.sample(range(dataset_size), args.num_samples)

    # ------------------------------------------------------------------
    # 3. Iterate over each model config
    # ------------------------------------------------------------------
    results = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file) as f:
                results = json.load(f)
        except Exception:
            pass

    for idx_config, config in enumerate(model_configs_to_run):
        model_identifier = config.get("model_identifier_for_results", config.get("decoder_name", f"model_{idx_config}").replace("/", "_"))
        logger.info("==== %s (%d/%d) ====", model_identifier, idx_config + 1, len(model_configs_to_run))

        # Skip if all selected samples already processed
        current_model_results = results.get(model_identifier, [])
        processed_image_paths = {r["image_path"] for r in current_model_results if "image_path" in r}
        actual_indices_for_dataloader = [i for i in sampled_indices if conceptual_dataset[i]["image_path"] not in processed_image_paths]
        if not actual_indices_for_dataloader:
            logger.info("All requested samples already processed for %s; skipping.", model_identifier)
            continue

        # ------------------------------------------------------------------
        # 3a. Load model + collator + dataloader
        # ------------------------------------------------------------------
        try:
            model_obj, processor, tokenizer = load_model_for_inference(config)
        except Exception as e:
            logger.error("Model load failed for %s: %s", model_identifier, e, exc_info=True)
            continue

        collator = MultimodalCollator(processor, tokenizer)
        subset = Subset(conceptual_dataset, actual_indices_for_dataloader)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

        # ------------------------------------------------------------------
        # 3b. NEW INNER LOOP – keeps alignment + metadata
        # ------------------------------------------------------------------
        num_items_processed_before = len(current_model_results)
        num_newly_processed_in_session = 0
        batch_counter = 0  # running offset inside subset

        for batch_data in tqdm(dataloader, desc=f"Generating with {model_identifier} (B={args.batch_size})"):
            try:
                # 1. Run model
                generated_captions = model_obj.generate_caption(batch_data)
                if isinstance(generated_captions, str):
                    generated_captions = [generated_captions]
                elif not isinstance(generated_captions, list):
                    logger.error("generate_caption returned %s", type(generated_captions))
                    generated_captions = []

                # 2. Build results list for this batch
                batch_results = []
                for i, caption in enumerate(generated_captions):
                    subset_idx = batch_counter + i
                    if subset_idx >= len(actual_indices_for_dataloader):
                        break
                    dataset_idx = actual_indices_for_dataloader[subset_idx]
                    item = conceptual_dataset[dataset_idx]
                    batch_results.append({
                        "image_path": item.get("image_path", f"unknown_{dataset_idx}"),
                        "base_caption": item.get("text", ""),
                        "generated_caption": caption
                    })

                current_model_results.extend(batch_results)
                num_newly_processed_in_session += len(batch_results)
                batch_counter += len(generated_captions)

                # 3. Periodic save ~ every 20 new items
                already_saved = len(current_model_results) - num_items_processed_before
                if args.save_after_each_model and already_saved > 0 and already_saved % 20 < args.batch_size:
                    results[model_identifier] = current_model_results
                    with open(args.output_file, "w") as f:
                        json.dump(results, f, indent=2)
                    logger.info("[intermediate] saved %d items for %s", len(current_model_results), model_identifier)

            except Exception as e_batch:
                logger.error("Batch failed for %s: %s", model_identifier, e_batch, exc_info=True)
                # Produce placeholder error entries to keep alignment
                for j in range(args.batch_size):
                    subset_idx = batch_counter + j
                    if subset_idx >= len(actual_indices_for_dataloader):
                        break
                    dataset_idx = actual_indices_for_dataloader[subset_idx]
                    try:
                        item = conceptual_dataset[dataset_idx]
                        image_path = item.get("image_path", f"unknown_{dataset_idx}")
                        base_caption = item.get("text", "")
                    except Exception:
                        image_path = f"error_accessing_item_{dataset_idx}"
                        base_caption = ""
                    current_model_results.append({
                        "image_path": image_path,
                        "base_caption": base_caption,
                        "generated_caption": f"ERROR: {str(e_batch)[:100]}..."
                    })
                batch_counter += args.batch_size

        # ------------------------------------------------------------------
        # 3c. Save after entire model is done
        # ------------------------------------------------------------------
        results[model_identifier] = current_model_results
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved %d total results for %s", len(current_model_results), model_identifier)

        # Cleanup GPU memory
        del model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)

    logger.info("All models complete – results in %s", args.output_file)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect() 
    main()


