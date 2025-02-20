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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from tqdm import tqdm
from pathlib import Path

from multimodel_utils import MultimodalModel, MultimodalCollator
from data import ConceptualCaptionsDataset

#unused but would help the automodelforcasuallm to register multimodality
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Load dataset from Hugging Face
data_dir = Path("./dataset/")
downloaded_indices = sorted([int(p.stem) for p in data_dir.glob("*.jpg") if p.stem.isdigit()])

dataset = load_dataset("google-research-datasets/conceptual_captions", split="train")
testing_indices = [0]
downloaded_subset = dataset.select(testing_indices)

# Define model pairs (processor name -> decoder config)
# Model configuration with identifiers and parameters
model_configs = [
    # BLIP
    {
        "processor_name": "Salesforce/blip-image-captioning-base",
        "decoder_class": BlipForConditionalGeneration,
        "decoder_name": "Salesforce/blip-image-captioning-base",
        "requires_original": True
    },

    # GIT-BART
    {
        "processor_name": "microsoft/git-base",
        "decoder_class": AutoModelForCausalLM,
        "decoder_name": "microsoft/git-base",
        "tokenizer_name": "microsoft/git-base"
    },

    {
        "processor_name": "nlpconnect/vit-gpt2-image-captioning",
        "decoder_class": VisionEncoderDecoderModel,
        "decoder_name": "nlpconnect/vit-gpt2-image-captioning",
        "processor_class": ViTImageProcessor
    },

    # CLIP-T5
    {
        "processor_name": "google/vit-base-patch16-224-in21k",
        "decoder_class": AutoModelForCausalLM,
        "decoder_name": "google-bert/bert-base-uncased",
        "tokenizer_name":"google-bert/bert-base-uncased"
    },

    # LLaVA
    {
        "processor_name": "xtuner/llava-llama-3-8b-v1_1-transformers",
        "decoder_class": LlamaForCausalLM,
        "decoder_name": "xtuner/llava-llama-3-8b-v1_1-transformers"
    },

    # Swin-GPT2
    {
        "processor_name": "microsoft/swin-base-patch4-window12-384",
        "decoder_class": AutoModelForCausalLM,
        "decoder_name": "google-bert/bert-base-uncased",
        "tokenizer_name": "google-bert/bert-base-uncased"
    },

    # Pix2Struct
    {
        "processor_name": "google/pix2struct-large",
        "decoder_class": Pix2StructForConditionalGeneration,
        "decoder_name": "google/pix2struct-large"
    },

    
    # Qwen-VL
    {
        "processor_name": "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
        "decoder_class": Qwen2VLForConditionalGeneration,
        "decoder_name": "Qwen/Qwen2-VL-7B-Instruct"
    },

    # DeepSeek Janus
    {
        "processor_name": "deepseek-ai/Janus-Pro-7B",
        "decoder_class": AutoModelForCausalLM,
        "decoder_name": "deepseek-ai/Janus-Pro-7B",
        "processor_class": VLChatProcessor,
        "decoder_kwargs": {"trust_remote_code": True}
    }
]
# Define Augmentations with Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomScale(scale_limit=0.2, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.CoarseDropout(num_holes_range=(4,8), fill="random", hole_height_range=(8,32), hole_width_range=(8,32), p=0.5)
])

def get_lora_config(model):
    """Smart LoRA configuration with fallback"""
    try:
        # First try automatic selection
        return LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules='all-linear',
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
    except ValueError:
        # Fallback to regex if automatic fails
        return LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=r'.*_proj$|.*query$|.*value$|.*embed|.*dense',
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

dataset = ConceptualCaptionsDataset(downloaded_subset, cache_dir=data_dir,transform=transform)

def select_best_model(processor, decoder, tokenizer, processor_name, decoder_name):
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
    vision_encoder_decoder_compatible = (
        # BartForConditionalGeneration,
        # T5ForConditionalGeneration,
        BertLMHeadModel,
        GPT2LMHeadModel,
        AutoModelForCausalLM,
    )
    # List of models that require their original implementation
    requires_original_implementation = (
        BlipForConditionalGeneration,
        Pix2StructForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
        LlamaForCausalLM,  
    )
    
    try:
        # Check if decoder is compatible with VisionEncoderDecoderModel
        if isinstance(decoder, vision_encoder_decoder_compatible):
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
            
            return model
            
        # Check if model requires original implementation
        elif isinstance(decoder, requires_original_implementation):
            return MultimodalModel(
                processor=processor,
                decoder=decoder,
                tokenizer=tokenizer
            )
        # ohh the irony of the models lol
        elif isinstance(decoder, (GitForCausalLM, VisionEncoderDecoderModel, LlamaForCausalLM)):
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

# Training Setup
def train_model(model_config, dataset):
    # Initialize model wrapper
    model = select_best_model(
        processor=model_config["processor"],
        decoder=model_config["decoder"],
        tokenizer=model_config["tokenizer"],
        processor_name=model_config["processor_name"],
        decoder_name=model_config["decoder_name"]
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"trainer_logs/{model_config['decoder_name']}",
        per_device_train_batch_size=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_dir='./logs',
        # save_strategy="epoch",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        save_safetensors=False,
        optim="paged_adamw_8bit"
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MultimodalCollator(
            processor=model_config["processor"],
            tokenizer=model_config["tokenizer"],
        ),
    )
    
    # Start training
    trainer.train()
    
    return model

# Main Training Loop
for idx, config in enumerate(model_configs):
    print("*"*100)
    print(f"Training model {idx+1}/{len(model_configs)}")
    print(f"training model pair: {config['processor_name']}, {config['decoder_name']}")
    print("-"*100)
    
    quantization_config = BitsAndBytesConfig(load_in_8_bit=True)
    # Dynamically load components
    processor = (config["processor_class"].from_pretrained(config["processor_name"])
                 if "processor_class" in config 
                 else AutoProcessor.from_pretrained(config["processor_name"]))

    model = config["decoder_class"].from_pretrained(
        config["decoder_name"],
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )

    #ps i dont usually write this ugly code
    try:
        tokenizer = (AutoTokenizer.from_pretrained(config["tokenizer_name"])
                     if "tokenizer_name" in config
                     else processor.tokenizer)
    except AttributeError as e:
        # I have you hugging face developers
        try:
            tokenizer = processor._tokenizer
        except AttributeError as e:
            tokenizer = AutoTokenizer.from_pretrained(config["processor_name"])

    if "Qwen" in config["processor_name"]:
        processor = AutoProcessor.from_pretrained(config['processor_name'], max_pixels=512*28*28)
        
    #LORA 
    lora_config = get_lora_config(model)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Train and save
    trained_model = train_model({
        "processor":processor,
        "decoder":model,
        "tokenizer": tokenizer,
        "requires_original": config.get("requires_original", False),
        "processor_name":config["processor_name"],
        "decoder_name":config["decoder_name"]
    }, dataset)
    
    output_dir = (
        f"./results/{config['processor_name'].replace('/','-')}_"
        f"{config['decoder_name'].replace('/','-')}"
    )
    # Save artifacts
    try:
        trained_model.decoder.save_pretrained(output_dir)
    except AttributeError as e:
        trained_model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)
    
    # Cleanup memory
    del processor, model, tokenizer, trained_model
    torch.cuda.empty_cache()
