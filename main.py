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

#unused but would help the automodelforcasuallm to register multimodality
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Load dataset from Hugging Face
data_dir = Path("./dataset/")

dataset = load_dataset("google-research-datasets/conceptual_captions", split="train")
downloaded_indices = sorted([int(p.stem) for p in data_dir.glob("*.jpg") if p.stem.isdigit()])

downloaded_indices_subset = random.sample(downloaded_indices, 75000) 

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

dataset = ConceptualCaptionsDataset(
    dataset,
    downloaded_indices_subset,
    cache_dir=data_dir,
    transform=transform
)
# Define model pairs (processor name -> decoder config)
# Model configuration with identifiers and parameters
model_configs = [
    # # BLIP
    # {
    #     "processor_name": "Salesforce/blip-image-captioning-base",
    #     "decoder_class": BlipForConditionalGeneration,
    #     "decoder_name": "Salesforce/blip-image-captioning-base",
    #     "requires_original": True,
    #     "batch_size":256
    # },
    #
    # # GIT
    # {
    #     "processor_name": "microsoft/git-base",
    #     "decoder_class": AutoModelForCausalLM,
    #     "decoder_name": "microsoft/git-base",
    #     "tokenizer_name": "microsoft/git-base",
    #     "batch_size":256
    # },

    # {
    #     "processor_name": "nlpconnect/vit-gpt2-image-captioning",
    #     "decoder_class": VisionEncoderDecoderModel,
    #     "decoder_name": "nlpconnect/vit-gpt2-image-captioning",
    #     "processor_class": ViTImageProcessor,
    #     "batch_size":256
    # },
    #
    # {
    #     "processor_name": "google/vit-base-patch16-224-in21k",
    #     "decoder_class": BertModel,
    #     "decoder_name": "google-bert/bert-base-uncased",
    #     "tokenizer_name":"google-bert/bert-base-uncased",
    #     "batch_size":256
    # },
    #
    #
    # # Swin-bert
    # {
    #     "processor_name": "microsoft/swin-base-patch4-window12-384",
    #     "decoder_class": BertModel,
    #     "decoder_name": "google-bert/bert-base-uncased",
    #     "tokenizer_name": "google-bert/bert-base-uncased",
    #     "batch_size":256
    # },
    #
    # # Pix2Struct
    # {
    #     "processor_name": "google/pix2struct-large",
    #     "decoder_class": Pix2StructForConditionalGeneration,
    #     "decoder_name": "google/pix2struct-large"
    # },

    # #LLAMA 
    # {
    #     "processor_name": "meta-llama/Llama-3.2-11B-Vision",
    #     "decoder_class": MllamaForConditionalGeneration,
    #     "decoder_name": "meta-llama/Llama-3.2-11B-Vision",
    #     "batch_size":128
    # },
    #
    # # Qwen-VL
    # {
    #     "processor_name": "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
    #     "decoder_class": Qwen2VLForConditionalGeneration,
    #     "decoder_name": "Qwen/Qwen2-VL-7B-Instruct",
    #     "batch_size":128
    # },
    #
    # # DeepSeek Janus
    {
        "processor_name": "deepseek-ai/Janus-Pro-7B",
        "decoder_class": AutoModelForCausalLM,
        "decoder_name": "deepseek-ai/Janus-Pro-7B",
        "processor_class": VLChatProcessor,
        "decoder_kwargs": {"trust_remote_code": True},
        "batch_size":128
    }
]

def get_lora_config(model):
    """
    Returns a LoRA configuration tailored to the underlying model's architecture.
    The function inspects the model's type (and optionally its config) and sets
    target_modules and task_type accordingly.
    """
    # Get a lowercase name for the model class.
    model_name = model.__class__.__name__.lower()
    
    # Default settings.
    r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    bias = "none"
    
    # Determine target modules and task type based on model type.
    if "blip" in model_name:
        # BLIP models (encoder-decoder with vision inputs)
        target_modules = "all-linear"
        task_type = "SEQ_2_SEQ_LM"
    elif "t5" in model_name or "bart" in model_name:
        # Encoder-decoder architectures.
        # For simplicity, here we target common attention projection modules.
        target_modules = ["q_proj", "v_proj"]
        task_type = "SEQ_2_SEQ_LM"
    elif "pix2struct" in model_name:
        target_modules = ["query",  "value"]
        task_type = "SEQ_2_SEQ_LM"
    elif "gpt2" in model_name or "llama" in model_name :
        # Decoder-only architectures
        target_modules = ["q_proj", "v_proj"]
        task_type = "CAUSAL_LM"
    elif "janus" in model_name or "qwen" in model_name:
        # Decoder-only architectures.
        target_modules =  ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
        task_type = "CAUSAL_LM"
    else:
        # Fallback using a regex pattern for linear layers.
        target_modules = r".*_proj$|.*query$|.*value$|.*dense"
        task_type = "CAUSAL_LM"
    
    # Create and return the LoRA configuration.
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )

# With custom preparation:
def prepare_janus_pro(model):
    # Manual gradient setup
    for param in model.parameters():
        param.requires_grad = False
        if param.dtype == torch.float32:
            param.requires_grad = True
    
    # Add missing embedding method
    if not hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings = lambda: model.decoder.get_input_embeddings()
    
    # Enable mixed precision
    # model.get_input_embeddings().weight.requires_grad = True
    return model


def select_best_model_inference(
    config
):
    # List of model types that work with VisionEncoderDecoderModel
    
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

    tokenizer = (AutoTokenizer.from_pretrained(config["tokenizer_name"])
                 if "tokenizer_name" in config
                 else AutoTokenizer.from_pretrained(config["processor_name"]))

    orig_instance = model.base_model.model if isinstance(model, PeftModel) else model
    # Check if decoder is compatible with VisionEncoderDecoderModel
    if isinstance(orig_instance, vision_encoder_decoder_compatible):
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            config["processor_name"],
            config["decoder_name"] 
        )
        # Set necessary config attributes
        model.config.decoder_start_token_id = (
            tokenizer.cls_token_id if tokenizer.cls_token_id 
            else tokenizer.bos_token_id
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        
        return model, processor, tokenizer

    # Check if model requires original implementation
    elif isinstance(orig_instance, requires_original_implementation):
        return MultimodalModel(
            processor=processor,
            decoder=model,
            tokenizer=tokenizer
        ), processor, tokenizer
    # ohh the irony of the models lol
    elif isinstance(orig_instance, (GitForCausalLM, VisionEncoderDecoderModel, LlamaForCausalLM)):
        return model , processor, tokenizer
    # For any other case, default to MultimodalModel
    else:
        return MultimodalModel(
            processor=processor,
            decoder=model,
            tokenizer=tokenizer
        ), processor, tokenizer

# After creating your model
def enable_grads(model):
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(True)
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))

# Training Setup
def train_model(model_config, dataset):
    # Initialize gradients on model
    model = model_config["decoder"]
    enable_grads(model)
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"trainer_logs/{model_config['processor_name']}",
        per_device_train_batch_size=model_config["batch_size"],
        auto_find_batch_size=True,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_dir=f'./logs_tf_board/{model_config["processor_name"]}',
        # save_strategy="epoch",
        save_steps=1500,
        resume_from_checkpoint=True,
        remove_unused_columns=False,
        # fp16=True,
        bf16=True,
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
        report_to="tensorboard",
        save_safetensors=False,
        optim="adafactor",
        gradient_checkpointing_kwargs={"use_reentrant":False},
        gradient_accumulation_steps=1,
        max_grad_norm=3.0,
        label_names=["labels"],
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
    print()
    print("*"*100)
    print(f"Training model {idx+1}/{len(model_configs)}")
    print(f"training model pair: {config['processor_name']}, {config['decoder_name']}")
    print("*"*100)
    print()
        
    
    model, processor, tokenizer = select_best_model(
        config
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Train and save
    trained_model = train_model({
        "processor":processor,
        "decoder":model,
        "tokenizer": tokenizer,
        "requires_original": config.get("requires_original", False),
        "processor_name":config["processor_name"],
        "decoder_name":config["decoder_name"],
        "batch_size": config["batch_size"]
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
