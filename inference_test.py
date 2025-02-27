import json
import random
import glob
import albumentations as A

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BlipForConditionalGeneration,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    VisionEncoderDecoderModel,
    Pix2StructForConditionalGeneration,
    GitForCausalLM,
    ViTImageProcessor,
    BertLMHeadModel,
    BertModel,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from tqdm import tqdm
from pathlib import Path

from multimodel_utils import MultimodalModel, MultimodalCollator
from data import ConceptualCaptionsDataset

from janus.models import MultiModalityCausalLM, VLChatProcessor

# Load dataset from Hugging Face
# data_dir = Path("./valid_dataset/")
#
# dataset = load_dataset("google-research-datasets/conceptual_captions", split="validation")
# downloaded_indices = sorted([int(p.stem) for p in data_dir.glob("*.jpg") if p.stem.isdigit()])
#
#
# # Define Augmentations with Albumentations
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
# ])
#
# dataset = ConceptualCaptionsDataset(
#     dataset,
#     downloaded_indices,
#     cache_dir=data_dir,
#     transform=transform
# )

model_configs = json.load(open("models.json"))

vision_encoder_decoder_compatible = (
    BertLMHeadModel,
    GPT2LMHeadModel,
    BertModel
)
# List of models that require their original implementation
requires_original_implementation = (
    BlipForConditionalGeneration,
    Pix2StructForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    LlamaForCausalLM,  
    MllamaForConditionalGeneration
)

def select_best_model_inference(
    processor,
    decoder,
    tokenizer,
    processor_name,
    decoder_name,
    peft_pretrained
):
    """
    Selects the appropriate model architecture based on encoder and decoder compatibility.
    
    Args:
        processor: The image processor/feature extractor
        decoder: The decoder model
        tokenizer: The tokenizer for text processing
    
    Returns:
        Either a VisionEncoderDecoderModel or MultimodalModel instance
    """
    # List of model types that work with VisionEncoderDecoderModel
    
    try:
        orig_instance = decoder.base_model.model if isinstance(decoder, PeftModel) else decoder
        # Check if decoder is compatible with VisionEncoderDecoderModel
        if isinstance(orig_instance, vision_encoder_decoder_compatible):
            model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                processor_name,
                decoder_name
            )
            # Set necessary config attributes
            model.config.decoder_start_token_id = (
                tokenizer.cls_token_id if tokenizer.cls_token_id 
                else tokenizer.bos_token_id
            )
            model.config.pad_token_id = tokenizer.pad_token_id
            
            if "pix2struct-large" not in config["processor_name"]:
                print()
                print(f"retrying to do peft for {config['processor_name']}")
                peft_model = PeftModel.from_pretrained(
                    model,
                    peft_pretrained
                )
                del decoder
                return peft_model
            
            return model

        # Check if model requires original implementation
        elif isinstance(orig_instance, requires_original_implementation):
            return MultimodalModel(
                processor=processor,
                decoder=decoder,
                tokenizer=tokenizer
            )
        # ohh the irony of the models lol
        elif isinstance(orig_instance, (GitForCausalLM, VisionEncoderDecoderModel, LlamaForCausalLM)):
            return decoder
        # For any other case, default to MultimodalModel
        else:
            return MultimodalModel(
                processor=processor,
                decoder=decoder,
                tokenizer=tokenizer
            )
            
    except Exception as e:
        raise ValueError(
            f"Failed to initialize model with processor {processor.__class__.__name__} "
            f"and decoder {decoder.__class__.__name__}: {str(e)}"
        )
for idx, config in enumerate(model_configs):
    if not config['peft']:
        continue
    print()
    print("*"*100)
    print(f"inferencing model {idx+1}/{len(model_configs)} {config['processor_name']}")
    print("*"*100)
    print()
    print(config["peft"])
    #
    quantization_config = BitsAndBytesConfig(load_in_8_bit=True) 
    # # Dynamically load components
    processor = (eval(config["processor_class"]).from_pretrained(config["processor_name"])
                 if "processor_class" in config 
                 else AutoProcessor.from_pretrained(config["processor_name"]))

    model = eval(config["decoder_class"]).from_pretrained(
        config["decoder_name"],
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )

    if not isinstance(model, vision_encoder_decoder_compatible):
        model = PeftModel.from_pretrained(
            model,
            config["peft"]
        )

    tokenizer = (AutoTokenizer.from_pretrained(config["tokenizer_name"])
                 if "tokenizer_name" in config
                 else AutoTokenizer.from_pretrained(config["processor_name"]))

    model = select_best_model_inference(
        processor,
        model,
        tokenizer,
        config["processor_name"],
        config["decoder_name"],
        config["peft"] 
    )
    if idx == 4:
        break


