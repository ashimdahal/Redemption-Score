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
    Qwen2_5_VLForConditionalGeneration
)
from peft import PeftModel
from PIL import Image
from janus.models import  VLChatProcessor, MultiModalityCausalLM
from transformers.data.data_collator import DataCollatorWithPadding
from janus.utils.io import load_pil_images
from qwen_vl_utils import process_vision_info
import torch


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
# Custom Data Collator to handle multimodal inputs
class MultimodalCollator(DataCollatorWithPadding):
    def __init__(self, processor, tokenizer ):
        super().__init__(tokenizer)
        self.processor = processor
        
        if isinstance(self.processor, MllamaProcessor):
            self.text_prompt = "<|image|> Give caption to this image like a normal human being. "

    def __call__(self, features):

        # Process text
        text = [item["text"] for item in features]
        #process Images
        images = [item["image"] for item in features]

        if isinstance(self.processor, Blip2Processor):
            processed_inputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                max_length=1024, # since its for both text and images it needs to be more
                padding="max_length"
            )
            labels = processed_inputs["input_ids"].clone()
            # Mask padding tokens.
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
                max_length=128 #since text tokens for captions arent large
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
            # if isinstance(self.processor, (
            #     Qwen2_5_VLForConditionalGeneration,
            #     Qwen2VLForConditionalGeneration
            # )
            #               ):
            #     images = [process_vision_info(example)[0] for example in features]

            processed_inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            labels = processed_inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            if isinstance(self.processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
                image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
            else:
                image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]  

            labels = processed_inputs["input_ids"].clone()
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            processed_inputs["labels"] = labels
            return processed_inputs

        elif isinstance(self.processor, VLChatProcessor):
            conversations = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\ncaption this image.",
                    "images": [image["image_path"]],
                }
                for image in features
            ]
            processed_inputs = self.processor(
                conversations=conversations,
                images=images, 
                paddint=True,
                return_tensors="pt",
                truncation=True,
                force_batchify=True
            )
        
            labels = processed_inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            processed_inputs["labels"] = labels
            return {
                "pixel_values":processed_inputs["pixel_values"],
                "labels":labels,
                # "attention_mask":processed_inputs["attention_mask"],
                "images_emb_mask":processed_inputs["images_emb_mask"],
                "images_seq_mask":processed_inputs["images_seq_mask"],
                "input_ids":processed_inputs["input_ids"],
                # "sft_format":processed_inputs["sft_format"]
            }
        elif (isinstance(self.processor, BlipProcessor)):
            processed_inputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding="max_length"
            )
            labels = processed_inputs["input_ids"].clone()
            # Mask padding tokens.
            processed_inputs["labels"] = labels

            return processed_inputs
        else:
            processed_inputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding="max_length"
            )

        labels = processed_inputs["input_ids"].clone() # Mask padding tokens.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": processed_inputs["pixel_values"],
            "input_ids": processed_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
            "label_names":labels
        }

# Custom Model Wrapper
class MultimodalModel(torch.nn.Module):
    def __init__(self, processor, decoder, tokenizer, freeze_vision_encoder=False):
        super().__init__()
        self.processor = processor
        self.decoder = decoder
        self.tokenizer = tokenizer
        
        self.orig_instance = self.decoder.base_model.model if isinstance(decoder, PeftModel) else self.decoder

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # Delegate to decoder's method if it exists
        if hasattr(self.orig_instance, 'gradient_checkpointing_enable'):
            return self.orig_instance.gradient_checkpointing_enable(*args, **kwargs)
        raise ValueError(f"model {self.orig_instance} doesnt have attribute gradient_checkpointing_enable")

    def forward(self, **kwargs):
        # Handle different model architectures
        if isinstance(self.orig_instance, (
            BlipForConditionalGeneration,
            BartForConditionalGeneration,
            Pix2StructForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
            MllamaForConditionalGeneration,
            Blip2ForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration
        )):
            # Encoder-decoder models
            outputs = self.decoder(
                **kwargs
            )
        elif isinstance(self.orig_instance, MultiModalityCausalLM):
            # the language model is a wrapper for llammaforcasualLM
            embeddings = self.decoder.prepare_inputs_embeds(**kwargs)
            outputs = self.decoder.language_model(
                inputs_embeds=embeddings,
                attention_mask=kwargs["attention_mask"],
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=40,
                # use_cache=True,
                labels=kwargs["labels"]
            )
        else:
            # fallback 
            inputs_embeds = self.decoder.get_input_embeddings()(kwargs["input_ids"])
            
            # Combine visual and text embeddings
            visual_features = self.decoder.vision_model(kwargs["pixel_values"]).last_hidden_state
            combined_embeds = torch.cat([visual_features, inputs_embeds], dim=1)
            
            outputs = self.decoder(
                **kwargs
            )
            
        return outputs

