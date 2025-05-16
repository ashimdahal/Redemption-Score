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
import albumentations as A

from data import CocoDataset
from evaluate_conceptual import load_model_for_inference
from multimodel_utils import MultimodalModel, MultimodalCollator 


def main():
    parser = argparse.ArgumentParser(description="Generate captions using trained models on COCO dataset")
    parser.add_argument(
        "--coco_images_dir",
        type=str,
        default="./coco_val2017_images/", # Example: where your 000000xxxxxx.jpg files are
        help="Directory containing COCO validation image files (e.g., 000000xxxxxx.jpg)."
    )
    parser.add_argument(
        "--coco_annotations_file",
        type=str,
        default="./coco_annotations/captions_val2017.json", # Example: path to your annotations
        help="Path to the COCO captions_val2017.json file."
    )
    parser.add_argument("--output_file", type=str, default="out_coco_direct.json", help="Output JSON file for COCO results.")
    parser.add_argument("--models_config", type=str, default="models.json", help="Path to models configuration JSON file.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of model indices to run (e.g., '0,2,3')")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--save_after_each_model", action="store_true", help="Save results after each model is processed")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3000, 
        help="Number of images to process from the COCO dataset (default: 3000). "
             "If 0, processes all valid images found by CocoDataset. "
             "The dataset will be formed from the first N valid images."
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for DataLoader (default: 5).")
    args = parser.parse_args()

    # ... (GLOBAL_DEVICE_PRIMARY, model_configs_all loading: same as before) ...
    global GLOBAL_DEVICE_PRIMARY 
    if args.cpu_only:
        logger.info("Forcing CPU usage as requested by --cpu_only flag.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        GLOBAL_DEVICE_PRIMARY = torch.device("cpu") 
        if torch.cuda.is_available(): 
            logger.warning("CUDA is still reported as available after setting CUDA_VISIBLE_DEVICES=''. Ensure PyTorch respects this.")
    else:
        GLOBAL_DEVICE_PRIMARY = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Global primary device set to: {GLOBAL_DEVICE_PRIMARY}")

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

    coco_images_path = Path(args.coco_images_dir)
    coco_annotations_path = Path(args.coco_annotations_file)

    logger.info(f"Initializing CocoDataset with images from: {coco_images_path} and annotations: {coco_annotations_path}")
    
    dataset_to_evaluate = None # This will be a Subset or the full CocoDataset

    try:
        transform = A.Compose([ A.LongestMaxSize(max_size=512), A.HorizontalFlip(p=0.5) ])
        
        # CocoDataset loads all valid items it finds.
        # The max_items_to_load can be used if CocoDataset itself should be limited.
        # Here, we let it load all, then take a subset for evaluation.
        full_coco_dataset = CocoDataset(
            images_base_dir=coco_images_path,
            annotations_json_file=coco_annotations_path,
            transform=transform
            # max_items_to_load=None # Load all valid items
        )
        
        if len(full_coco_dataset) == 0:
            logger.error(f"CocoDataset is empty. Check paths and annotation file contents.")
            return
        logger.info(f"Initialized CocoDataset with {len(full_coco_dataset)} total valid image-caption pairs.")

        # Determine the subset of data to use for evaluation based on args.num_samples
        num_available_samples = len(full_coco_dataset)
        
        if args.num_samples == 0: # User wants all available samples
            num_to_use_for_this_run = num_available_samples
            logger.info(f"Processing all {num_available_samples} available samples from CocoDataset (--num_samples is 0).")
            dataset_to_evaluate = full_coco_dataset
        elif args.num_samples > 0:
            if args.num_samples <= num_available_samples:
                num_to_use_for_this_run = args.num_samples
                logger.info(f"Processing the first {num_to_use_for_this_run} samples from CocoDataset.")
                dataset_to_evaluate = Subset(full_coco_dataset, list(range(num_to_use_for_this_run)))
            else: # args.num_samples > num_available_samples
                num_to_use_for_this_run = num_available_samples
                logger.warning(f"Requested {args.num_samples} samples, but only {num_available_samples} are available in CocoDataset. Processing all available.")
                dataset_to_evaluate = full_coco_dataset
        
        if not dataset_to_evaluate or len(dataset_to_evaluate) == 0:
             logger.error("No samples selected or available for evaluation.")
             return
        
        logger.info(f"Effective number of samples for this evaluation run: {len(dataset_to_evaluate)}")


    except Exception as e:
        logger.error(f"Error initializing or subsetting CocoDataset: {e}"); traceback.print_exc(); return

    # The rest of the script (results loading, model iteration loop) remains largely the same,
    # as it operates on `dataset_to_evaluate` (which is either CocoDataset or a Subset of it).

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
        
        current_model_results_list = results.get(model_identifier, [])
        processed_image_paths_for_this_model = {entry["image_path"] for entry in current_model_results_list if "image_path" in entry}
        
        model_obj = None 
        
        try:
            # `dataset_to_evaluate` is already the correct set of samples for this run.
            # We need to filter these based on `processed_image_paths_for_this_model`.
            
            # If dataset_to_evaluate is a Subset, its indices refer to the original full_coco_dataset.
            # If dataset_to_evaluate is full_coco_dataset, indices are direct.
            
            indices_for_dataloader_for_this_model_run = [] # These will be indices into dataset_to_evaluate
            
            for i in range(len(dataset_to_evaluate)):
                # To get the image path, we need to access the item from dataset_to_evaluate
                # This involves calling __getitem__ if we don't have a direct get_image_path for Subset.
                # A bit inefficient to call __getitem__ just for path, but robust.
                # Or, if dataset_to_evaluate is Subset, get original index and use full_coco_dataset.get_image_path()
                
                item_path = ""
                if isinstance(dataset_to_evaluate, Subset):
                    original_dataset_idx = dataset_to_evaluate.indices[i]
                    item_path = dataset_to_evaluate.dataset.get_image_path(original_dataset_idx) # Access underlying dataset's method
                else: # It's the full_coco_dataset
                    item_path = dataset_to_evaluate.get_image_path(i)

                if item_path not in processed_image_paths_for_this_model:
                    indices_for_dataloader_for_this_model_run.append(i) # This index 'i' is for dataset_to_evaluate
            
            if not indices_for_dataloader_for_this_model_run:
                logger.info(f"All {len(dataset_to_evaluate)} selected samples for {model_identifier} already processed. Skipping model.")
                continue
            
            logger.info(f"Number of new items to process for {model_identifier}: {len(indices_for_dataloader_for_this_model_run)}")

            # Create a new Subset using these new indices, now relative to `dataset_to_evaluate`
            final_subset_for_model_dataloader = Subset(dataset_to_evaluate, indices_for_dataloader_for_this_model_run)
            
            model_obj, processor, tokenizer = load_model_for_inference(config) 

            collator = MultimodalCollator(processor, tokenizer) # Ensure this is defined
            dataloader = DataLoader(
                final_subset_for_model_dataloader, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=0,
                collate_fn=collator
            ) 
            
            # ... (The rest of the batch processing, result saving, and cleanup loop is identical to the previous `evaluate_coco.py`) ...
            num_newly_processed_in_this_session_for_model = 0
            for batch_data in tqdm(dataloader, desc=f"Generating with {model_identifier} (Batch Size: {args.batch_size})"):
                batch_output_results = [] 
                try:
                    generated_batch_items = model_obj.generate_caption(batch_data)
                    if generated_batch_items and isinstance(generated_batch_items, list):
                        batch_output_results.extend(generated_batch_items)
                        num_newly_processed_in_this_session_for_model += len(generated_batch_items)
                    else:
                        logger.error(f"model.generate_caption did not return a list. Received: {type(generated_batch_items)}. Marking batch items as failed.")
                        original_paths = batch_data.get('image_paths_original', [f"unknown_path_batch_fail_{i}" for i in range(len(batch_data.get('pixel_values', [])))])
                        original_texts = batch_data.get('texts_original', [""] * len(original_paths))
                        for i_fail in range(len(original_paths)):
                             batch_output_results.append({
                                "image_path": original_paths[i_fail],
                                "base_caption": original_texts[i_fail],
                                "generated_caption": "ERROR: Batch processing failed or returned invalid format."
                            })
                except Exception as e_batch_call: 
                    logger.error(f"Error in model.generate_caption for a batch with {model_identifier}: {e_batch_call}")
                    traceback.print_exc()
                    original_paths = batch_data.get('image_paths_original', [f"unknown_path_batch_exception_{i}" for i in range(len(batch_data.get('pixel_values', [])))])
                    original_texts = batch_data.get('texts_original', [""] * len(original_paths))
                    for i_ex in range(len(original_paths)):
                        batch_output_results.append({
                            "image_path": original_paths[i_ex],
                            "base_caption": original_texts[i_ex],
                            "generated_caption": f"ERROR_CALLING_BATCH_GENERATE: {str(e_batch_call)[:100]}..."
                        })
                current_model_results_list.extend(batch_output_results)
                if args.save_after_each_model and batch_output_results: 
                    items_per_intermediate_save = 20 
                    is_last_batch_for_model = (num_newly_processed_in_this_session_for_model == len(final_subset_for_model_dataloader)) # Compare with length of current dataloader's source
                    if (num_newly_processed_in_this_session_for_model % items_per_intermediate_save < args.batch_size and \
                        num_newly_processed_in_this_session_for_model // items_per_intermediate_save > \
                        (num_newly_processed_in_this_session_for_model - len(batch_output_results)) // items_per_intermediate_save) \
                        or is_last_batch_for_model:
                        results[model_identifier] = current_model_results_list 
                        with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
                        logger.info(f"Saved intermediate results ({len(current_model_results_list)} total for model, {num_newly_processed_in_this_session_for_model} new in this session) to '{args.output_file}'")

            results[model_identifier] = current_model_results_list 
            if args.save_after_each_model and num_newly_processed_in_this_session_for_model > 0 : 
                with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
                logger.info(f"Saved results after processing all batches for {model_identifier} ({len(current_model_results_list)} total items) to '{args.output_file}'")
        
        except Exception as e_model_load_or_process: 
            logger.error(f"FATAL: Error during setup or processing for model {model_identifier}: {e_model_load_or_process}")
            traceback.print_exc()
            error_entry = {"error_message": f"Model {model_identifier} failed.", "details": f"{str(e_model_load_or_process)[:200]}..."}
            if model_identifier not in results:
                results[model_identifier] = [error_entry]
            elif not any(item.get("error_message") == error_entry["error_message"] for item in results[model_identifier]):
                results[model_identifier].append(error_entry)
            if args.save_after_each_model: 
                with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
        finally: 
            if model_obj is not None:
                logger.debug(f"Deleting model object for {model_identifier} to free memory.")
                del model_obj
                if 'processor' in locals(): del processor 
                if 'tokenizer' in locals(): del tokenizer
            if GLOBAL_DEVICE_PRIMARY.type == 'cuda':
                logger.debug(f"Clearing CUDA cache after model {model_identifier}.")
                torch.cuda.empty_cache(); gc.collect(); time.sleep(0.5) 

    try:
        with open(args.output_file, 'w') as f: json.dump(results, f, indent=2)
        logger.info(f"All processing complete. Final results saved to '{args.output_file}'")
    except Exception as e_final_save:
        logger.error(f"Error during final save to '{args.output_file}': {e_final_save}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect() 
    main()

