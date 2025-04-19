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
    MllamaForConditionalGeneration,
    SwinModel
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import evaluate

from tqdm import tqdm
from pathlib import Path

from multimodel_utils import MultimodalModel, MultimodalCollator
from data import ConceptualCaptionsDataset

from janus.models import MultiModalityCausalLM, VLChatProcessor

# Load dataset from Hugging Face
data_dir = Path("./valid_dataset/")


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

@torch.no_grad()
@torch.inference_mode()
def evaluate_model(
    model,
    processor,
    tokenizer,
    config,
    batch_size=2,
    save_samples=30,
    device=torch.device("cuda")
):
    if isinstance(model, MultimodalModel) and isinstance(model.decoder, MultiModalityCausalLM):
        num_samples=50
    else:
        num_samples=50
    dataset = load_dataset("google-research-datasets/conceptual_captions", split="validation")
    downloaded_indices = sorted([int(p.stem) for p in data_dir.glob("*.jpg") if p.stem.isdigit()])

    downloaded_indices = downloaded_indices[:num_samples]

# Define Augmentations with Albumentations
    transform = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.HorizontalFlip(p=0.5),
    ])

    dataset = ConceptualCaptionsDataset(
        dataset,
        downloaded_indices,
        cache_dir=data_dir,
        transform=transform
    )

    print(f"making dataloader with {batch_size} batches")
    collator = MultimodalCollator(processor, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    model.to(device)
    model.eval()

    all_preds = []
    all_references = []

    save_indices = random.sample(range(len(dataset)), min(save_samples, len(dataset)))

    samples_to_save = []

    
    if isinstance(model, MultimodalModel) and isinstance(model.decoder, MultiModalityCausalLM):

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator
        )
    print(f"running inferences on the dataset")
    with torch.no_grad():
        for i,batch in enumerate(tqdm(dataloader)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            if "labels" in batch:
                labels = batch.pop("labels")
            if isinstance(model, MultimodalModel):
                if isinstance(model.decoder, MllamaForConditionalGeneration):
                    generated_ids = model.decoder.generate(**batch, max_new_tokens=50)
                    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)]
                elif isinstance(model.decoder, Qwen2VLForConditionalGeneration):
                    generated_ids = model.decoder.generate(**batch, max_new_tokens=50)
                    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)]
                    labels[labels==-100] = 0
                elif isinstance(model.decoder, MultiModalityCausalLM):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        input_embeds = model.decoder.prepare_inputs_embeds(**batch)
                        generated_ids = model.decoder.language_model.generate(
                            inputs_embeds=input_embeds,
                            # attention_mask=batch["attention_mask"],
                            pad_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=512,
                        )
                else:
                    generated_ids = model.decoder.generate(pixel_values = batch["pixel_values"])
            else:
                model.generation_config.decoder_start_token_id = (
                    tokenizer.cls_token_id if tokenizer.cls_token_id 
                    else tokenizer.bos_token_id
                )
                generated_ids = model.generate(pixel_values = batch["pixel_values"])
                labels[labels==-100] = 0

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(preds)
            all_references.extend(refs)

            batch_start_idx = i * batch_size
            for j in range(len(preds)):
                sample_idx = batch_start_idx + j
                if sample_idx in save_indices:
                    # Get the image path
                    img_path = dataset.get_image_path(sample_idx)
                    
                    samples_to_save.append({
                        "image_path": str(img_path),
                        "reference": refs[j],
                        "prediction": preds[j]
                    })

        print(all_preds)
        print(all_references)
        print("calculating metrics")
        metrics = calculate_metrics_hf(all_references, all_preds)

        output_dir = Path(
            f"./evaluation_results/{config['processor_name'].replace('/','-')}_"
            f"{config['decoder_name'].replace('/','-')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics , f, indent=4)

        with open(output_dir / "samples.json", "w")as f:
            json.dump(samples_to_save, f, indent=4)

        print(f"evaluation completed and saved to {output_dir}")
            

def calculate_metrics_hf(references, predictions):
    """
    Calculate metrics using HuggingFace's evaluate library
    """
    # Load metrics
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    google_bleu = evaluate.load("google_bleu")

    # BLEU scores
    bleu_results = bleu.compute(predictions=predictions, references=[[r] for r in references])

    # METEOR score
    meteor_results = meteor.compute(predictions=predictions, references=references)

    # BERTScore
    bertscore_results = bertscore.compute(
        predictions=predictions, 
        references=references, 
        lang="en", 
        rescale_with_baseline=True
    )

    rougescore_results = rouge.compute(predictions=predictions, references=references)
    google_bleu_score = google_bleu.compute(predictions=predictions, references=references)
    return {
        "Bleu": bleu_results,
        "meteor":meteor_results,
        "bertscore": bertscore_results,
        "rouge":rougescore_results,
        "google_bleu":google_bleu_score
    }

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
            
def main():
    for idx, config in enumerate(model_configs):
        print()
        print("*"*100)
        print(f"inferencing model {idx+1}/{len(model_configs)} {config['processor_name']}")
        print("*"*100)
        print()
        

        model, processor, tokenizer = select_best_model_inference(
            config
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model._resize_token_embeddings(len(tokenizer))

        model.eval()

        evaluate_model(
            model,
            processor,
            tokenizer,
            config,
        )

if __name__ == "__main__":
    main()
