import glob
import albumentations as A
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BlipForConditionalGeneration,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    Qwen2ForCausalLM
)

from datasets import load_dataset

# Set a writable cache directory
from tqdm import tqdm

from multimodel_utils import MultimodalModel, MultimodalCollator
from data import ConceptualCaptionsDataset

# Load dataset from Hugging Face
data_dir = Path("./dataset/")
downloaded_indices = sorted([int(p.stem) for p in data_dir.glob("*.jpg") if p.stem.isdigit()])

dataset = load_dataset("google-research-datasets/conceptual_captions", split="train")
downloaded_subset = dataset.select(downloaded_indices)

processors = [
    AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
    AutoProcessor.from_pretrained("microsoft/git-base"),
    AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
    AutoProcessor.from_pretrained("openai/clip-vit-large-patch14"),
    AutoProcessor.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers"),
    AutoProcessor.from_pretrained("google/pix2struct-large"),
    AutoProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384"),
    AutoProcessor.from_pretrained("Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"),
    AutoProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B"),
]


# Define model pairs (processor name -> decoder config)
model_pairs = [
    # BLIP - Requires original implementation
    {
        "processor": "Salesforce/blip-image-captioning-base",
        "decoder": BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ),
        "requires_original": True
    },
    
    # GIT - Paired with BART for seq2seq capability
    {
        "processor": "microsoft/git-base",
        "decoder": BartForConditionalGeneration.from_pretrained("facebook/bart-base"),
        "tokenizer": AutoProcessor.from_pretrained("facebook/bart-base").tokenizer
    },
    
    # ViT-GPT2 - Original pairing works best
    {
        "processor": "nlpconnect/vit-gpt2-image-captioning",
        "decoder": GPT2LMHeadModel.from_pretrained("gpt2-medium")
    },
    
    # CLIP - Paired with T5 for text generation
    {
        "processor": "openai/clip-vit-large-patch14",
        "decoder": T5ForConditionalGeneration.from_pretrained("t5-large"),
        "tokenizer": AutoProcessor.from_pretrained("t5-large").tokenizer
    },
    
    # LLaVA - Use original Llama architecture
    {
        "processor": "xtuner/llava-llama-3-8b-v1_1-transformers",
        "decoder": LlamaForCausalLM.from_pretrained(
            "xtuner/llava-llama-3-8b-v1_1-transformers",
            trust_remote_code=True
        )
    },
    
    # Pix2Struct - Keep original architecture
    {
        "processor": "google/pix2struct-large",
        "decoder": T5ForConditionalGeneration.from_pretrained("google/pix2struct-large")
    },
    
    # Swin Transformer - Paired with GPT-2
    {
        "processor": "microsoft/swin-base-patch4-window12-384",
        "decoder": GPT2LMHeadModel.from_pretrained("gpt2-xl"),
        "tokenizer": AutoProcessor.from_pretrained("gpt2-xl").tokenizer
    },
    
    # Qwen-VL - Use original model
    {
        "processor": "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
        "decoder": Qwen2ForCausalLM.from_pretrained(
            "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
            trust_remote_code=True
        )
    },
    
    # DeepSeek Janus - Use as causal LM
    {
        "processor": "deepseek-ai/Janus-Pro-7B",
        "decoder": AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/Janus-Pro-7B",
            trust_remote_code=True
        )
    }
]


# Create final model list with processors and decoders
models = []
for pair in model_pairs:
    processor = next(p for p in processors if p.name_or_path == pair["processor"])
    
    model_config = {
        "processor": processor,
        "decoder": pair["decoder"],
        "requires_original": pair.get("requires_original", False)
    }
    
    # Handle separate tokenizers where needed
    if "tokenizer" in pair:
        model_config["tokenizer"] = pair["tokenizer"]
    else:
        model_config["tokenizer"] = processor.tokenizer
        
    models.append(model_config)

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

dataset = ConceptualCaptionsDataset(downloaded_subset, transform=transform)

# Training Setup
def train_model(model_config, dataset):
    # Initialize model wrapper
    # to be tested and debugged but we use predefined visionencoderdecodermodel from hf
    # model = MultimodalModel(
    #     processor=model_config["processor"],
    #     decoder=model_config["decoder"],
    #     tokenizer=model_config["tokenizer"]
    # )

    model = VisionEncoderDecoderModel(
        encoder=model_config["processor"],
        decoder=model_config["decoder"]
    )
    
    tokenizer = model_config["tokenizer"]
    model.config.decoder_start_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id else tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_config['processor'].name_or_path}_{model_config["decoder"].name_or_path}",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_dir='./logs',
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
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
for idx, model_config in enumerate(models):
    print(f"Training model {idx+1}/{len(models)}: {model_config['processor'].name_or_path}")
    
    # Train model
    trained_model = train_model(model_config, dataset)
    
    # Save model
    save_path = f"./saved_models/{model_config['processor'].name_or_path}"
    trained_model.decoder.save_pretrained(save_path)
    model_config["tokenizer"].save_pretrained(save_path)
