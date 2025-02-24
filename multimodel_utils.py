from transformers import (
    BlipForConditionalGeneration,
    BartForConditionalGeneration,
    ViTImageProcessor,
    LlamaForCausalLM,
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    MllamaProcessor,
    MllamaForConditionalGeneration,
    BlipProcessor
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
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": "describe the image",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}],
        },
    ]
# Custom Data Collator to handle multimodal inputs
class MultimodalCollator(DataCollatorWithPadding):
    def __init__(self, processor, tokenizer ):
        super().__init__(tokenizer)
        self.processor = processor
        
        if isinstance(self.processor, MllamaProcessor):
            self.text_prompt = "<|image|> Describe this image."

    def __call__(self, features):
        # Process text
        text = [item["text"] for item in features]
        text_inputs = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )

        # Process images
        images = [item["image"] for item in features]
        if isinstance(self.processor, ViTImageProcessor):
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
            return{
                "pixel_values": pixel_values,
                "labels":text_inputs["input_ids"],
            }

        elif isinstance(self.processor, Pix2StructProcessor):
            headers = ["Describe the contents of this image"] * len(images)
            processed_outputs = self.processor(images=images, text=headers, return_tensors="pt")

            return {
                "flattened_patches":processed_outputs["flattened_patches"],
                "attention_mask":processed_outputs["attention_mask"],
                # "decoder_input_ids":text_inputs["input_ids"]
                "labels": text_inputs["input_ids"]  # For causal LM models
            } 
        elif isinstance(self.processor, (Qwen2VLProcessor)):
            features = [format_data(feature) for feature in features]
            texts = [self.processor.apply_chat_template(example, tokenize=False) for example in features]

            image_inputs = [process_vision_info(example)[0] for example in features]
            processed_outputs = self.processor(
                text=texts,
                images=image_inputs,
                return_tensors="pt",
                padding=True
            )
            labels = processed_outputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            image_tokens = [151652, 151653, 151655]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            processed_outputs["labels"] = labels
            return processed_outputs

        elif isinstance(self.processor, MllamaProcessor):
            #needs explicitly set same token size on input and output
            processed_outputs = self.processor(
                images=images,
                text=[self.text_prompt] * len(images),
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            processed_outputs["labels"] = text_inputs["input_ids"]
            return processed_outputs
        elif isinstance(self.processor, VLChatProcessor):
            conversations = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\nDescribe this image.",
                    "images": [image],
                }
                for image in images
            ]
            processed_outputs = self.processor(
                conversations=conversations,
                images=images, 
                force_batchify=True
            )
            labels = processed_outputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            processed_outputs["labels"] = labels
            return {
                "pixel_values":processed_outputs["pixel_values"],
                "labels":processed_outputs["input_ids"],
                "attention_mask":processed_outputs["attention_mask"],
                "images_emb_mask":processed_outputs["images_emb_mask"],
                "images_seq_mask":processed_outputs["images_seq_mask"],
                "input_ids":processed_outputs["input_ids"]
            }
        elif (isinstance(self.processor, BlipProcessor)):
            processed_outputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding="max_length"
            )
            labels = processed_outputs["input_ids"].clone()
            # Mask padding tokens.
            processed_outputs["labels"] = labels

            return processed_outputs
        else:
            processed_outputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding="max_length"
            )

        print(self.processor)
        labels = processed_outputs["input_ids"].clone() # Mask padding tokens.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": processed_outputs["pixel_values"],
            "input_ids": processed_outputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels
        }

# Custom Model Wrapper
class MultimodalModel(torch.nn.Module):
    def __init__(self, processor, decoder, tokenizer, freeze_vision_encoder=False):
        super().__init__()
        self.processor = processor
        self.decoder = decoder
        self.tokenizer = tokenizer
        
        self.has_vision_encoder = hasattr(self.decoder, "vision_model") or hasattr(self.decoder, "encoder")

        self.orig_instance = self.decoder.base_model.model if isinstance(decoder, PeftModel) else self.decoder
        # Freeze vision encoder if required
        if freeze_vision_encoder and self.has_vision_encoder:
            vision_encoder = getattr(self.decoder, "vision_model", None) or getattr(self.decoder, "encoder", None)
            if vision_encoder:
                for param in vision_encoder.parameters():
                    param.requires_grad = False

    def forward(self, **kwargs):
        # Handle different model architectures
        if isinstance(self.orig_instance, (
            BlipForConditionalGeneration,
            LlamaForCausalLM,
            BartForConditionalGeneration,
            Pix2StructForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
            MllamaForConditionalGeneration,
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
            print(self.orig_instance)
            inputs_embeds = self.decoder.get_input_embeddings()(kwargs["input_ids"])
            
            # Combine visual and text embeddings
            visual_features = self.decoder.vision_model(kwargs["pixel_values"]).last_hidden_state
            combined_embeds = torch.cat([visual_features, inputs_embeds], dim=1)
            
            outputs = self.decoder(
                **kwargs
            )
            
        return outputs

