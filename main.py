import glob
import albumentations as A

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import (
    AutoModel,
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
    Pix2StructForConditionalGeneration
)

from datasets import load_dataset

from tqdm import tqdm
from pathlib import Path

from multimodel_utils import MultimodalModel, MultimodalCollator
from data import ConceptualCaptionsDataset

#unused but would help the automodelforcasuallm to register multimodality
from janus.models import MultiModalityCausalLM

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
        "decoder_class": BartForConditionalGeneration,
        "decoder_name": "facebook/bart-base",
        "tokenizer_name": "facebook/bart-base"
    },
    
    # ViT-GPT2
    {
        "processor_name": "nlpconnect/vit-gpt2-image-captioning",
        "decoder_class": GPT2LMHeadModel,
        "decoder_name": "gpt2-medium"
    },
    
    # CLIP-T5
    {
        "processor_name": "openai/clip-vit-large-patch14",
        "decoder_class": T5ForConditionalGeneration,
        "decoder_name": "t5-large",
        "tokenizer_name": "t5-large"
    },
    
    # LLaVA
    {
        "processor_name": "xtuner/llava-llama-3-8b-v1_1-transformers",
        "decoder_class": LlamaForCausalLM,
        "decoder_name": "xtuner/llava-llama-3-8b-v1_1-transformers"
    },
    
    # Pix2Struct
    {
        "processor_name": "google/pix2struct-large",
        "decoder_class": Pix2StructForConditionalGeneration,
        "decoder_name": "google/pix2struct-large"
    },
    
    # Swin-GPT2
    {
        "processor_name": "microsoft/swin-base-patch4-window12-384",
        "decoder_class": GPT2LMHeadModel,
        "decoder_name": "gpt2-xl",
        "tokenizer_name": "gpt2-xl"
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

dataset = ConceptualCaptionsDataset(downloaded_subset, cache_dir=data_dir,transform=transform)

# Training Setup
def train_model(model_config, dataset):
    # Initialize model wrapper
    # to be tested and debugged but we use predefined visionencoderdecodermodel from hf
    # model = MultimodalModel(
    #     processor=model_config["processor"],
    #     decoder=model_config["decoder"],
    #     tokenizer=model_config["tokenizer"]
    # )
    model = select_best_model(decoder, processor, tokenizer)

    output_dir = (
        f"./results/{model_config['processor'].name_or_path.replace('/','-')}_"
        f"{model_config['decoder'].name_or_path.replace('/','-')}"
    )
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_dir='./logs',
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none"
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MultimodalCollator(
            processor=model_config["processor"],
            tokenizer=model_config["tokenizer"]
        ),
    )
    
    # Start training
    trainer.train()
    
    return model

# Main Training Loop
for idx, config in enumerate(model_configs):
    print(f"Training model {idx+1}/{len(model_configs)}")
    print(f"training model pair: {config['processor_name']}, {config['decoder_name']}")
    
    # Dynamically load components
    processor = AutoProcessor.from_pretrained(config["processor_name"])
    model = config["decoder_class"].from_pretrained(config["decoder_name"])
    
    tokenizer = (AutoTokenizer.from_pretrained(config["tokenizer_name"])
                 if "tokenizer_name" in config
                 else processor.tokenizer)

    # Train and save
    trained_model = train_model({
        "processor":processor,
        "model":model,
        "tokenizer": tokenizer,
        "requires_original": config.get("requires_original", False),
    }, dataset)
    
    # Save artifacts
    save_path = f"./saved_models/{config['processor_name'].replace('/','-')}"
    trained_model.decoder.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Cleanup memory
    del processor, decoder, tokenizer, trained_model
    torch.cuda.empty_cache()
