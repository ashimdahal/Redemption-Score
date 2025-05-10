import torch
from PIL import Image
import logging 
import gc 

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BartForConditionalGeneration, 
    ViTImageProcessor,
    Pix2StructProcessor, 
    Pix2StructForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLProcessor, 
    MllamaProcessor,
    MllamaForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Qwen2VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    VisionEncoderDecoderModel,
    GitForCausalLM,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from peft import PeftModel
from transformers.data.data_collator import DataCollatorWithPadding

from qwen_vl_utils import process_vision_info # Restoring user's import

logger = logging.getLogger(__name__) 

system_message = """You are a Vision Language Model specialized in captioning or providing a short description of them"""

def format_data(sample):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image_path"],
                },
                {
                    "type": "text",
                    "text": "Give caption to the image",
                },
            ],
        },
    ]

class MultimodalCollator(DataCollatorWithPadding):
    def __init__(self, processor, tokenizer ):
        super().__init__(tokenizer)
        self.processor = processor
        
        if isinstance(self.processor, MllamaProcessor):
            self.text_prompt = "<|image|> Give caption to this image like a normal human being. "

    def __call__(self, features):
        text = [item["text"] for item in features]
        images = [item["image"] for item in features] 

        if isinstance(self.processor, Blip2Processor):
            processed_inputs = self.processor(
                images=images, text=text, return_tensors="pt",
                padding="max_length", truncation=True, 
                max_length=1024 
            )
            labels = processed_inputs["input_ids"].clone()
            processed_inputs["labels"] = labels 
            return processed_inputs

        text_inputs = self.tokenizer( 
            text, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )

        if isinstance(self.processor, ViTImageProcessor):
            text_inputs = self.tokenizer( 
                text, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt",
                max_length=128 
            )
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
            return{ 
                "pixel_values": pixel_values,
                "labels":text_inputs["input_ids"],
                "label_names":text_inputs["input_ids"],
            }
        
        elif isinstance(self.processor, (
                Qwen2VLProcessor,
                Qwen2_5_VLProcessor,
                MllamaProcessor
            )):
            features = [format_data(feature) for feature in features]
            texts = [
                self.processor.apply_chat_template(
                    example,
                    tokenize=False,
                    add_generation_prompt=True
                ) for example in features
            ]
            if isinstance(self.processor, ( 
                Qwen2_5_VLForConditionalGeneration,
                Qwen2VLForConditionalGeneration
            )
                            ):
                images = [process_vision_info(example)[0] for example in features]

            processed_inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            labels = processed_inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            if isinstance(self.processor, Qwen2VLProcessor):  
                image_tokens = [151652, 151653, 151655]  
            else:
                image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)] 

            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            processed_inputs["labels"] = labels

            return processed_inputs
        
        elif (isinstance(self.processor, BlipProcessor)): 
            processed_inputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding="max_length"
            )
            labels = processed_inputs["input_ids"].clone()
            processed_inputs["labels"] = labels
            return processed_inputs
        
        else: 
            processed_inputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding="max_length"
            )
            labels = processed_inputs["input_ids"].clone() 
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            return {
                "pixel_values": processed_inputs["pixel_values"],
                "input_ids": processed_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "labels": labels,
                "label_names":labels
            }

class MultimodalModel(torch.nn.Module):
    def __init__(self, processor, decoder, tokenizer, freeze_vision_encoder=False):
        super().__init__()
        self.processor = processor
        self.decoder = decoder 
        self.tokenizer = tokenizer
        
        if isinstance(self.decoder, PeftModel):
            self.orig_instance = self.decoder.base_model.model
        else:
            self.orig_instance = self.decoder
        
    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.orig_instance, 'gradient_checkpointing_enable'):
            return self.orig_instance.gradient_checkpointing_enable(*args, **kwargs)
        elif hasattr(self.decoder, 'gradient_checkpointing_enable') and self.decoder is not self.orig_instance:
             return self.decoder.gradient_checkpointing_enable(*args, **kwargs)
        raise ValueError(f"model {self.orig_instance} doesnt have attribute gradient_checkpointing_enable")

    def forward(self, **kwargs):
        outputs = self.decoder(**kwargs)
        return outputs
    
    def generate_caption(self, batch):
        """Generate caption for an image with better error handling and dynamic token management.
        Returns:
            Generated caption string
        """
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.decoder.device)

        if "labels" in batch:
            labels = batch.pop("labels")

        if isinstance(self.orig_instance, (
            MllamaForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration
        )):
            generated_ids = self.decoder.generate(**batch, max_new_tokens=50)
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)]

        elif isinstance(self.orig_instance, (
            BertLMHeadModel,
            GPT2LMHeadModel,
            BertModel 
        )):
            self.decoder.generation_config.decoder_start_token_id = (
                self.tokenizer.cls_token_id if self.tokenizer.cls_token_id 
                else self.tokenizer.bos_token_id
            )
            generated_ids = self.decoder.generate(pixel_values = batch["pixel_values"])
            labels[labels==-100] = 0
        else:
            generated_ids = model.decoder.generate(pixel_values = batch["pixel_values"])
        
        preds = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return preds


