from transformers import (
    BlipForConditionalGeneration,
    BartForConditionalGeneration,
    ViTImageProcessor,
    LlamaForCausalLM,
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration
)
from janus.models import  VLChatProcessor, MultiModalityCausalLM
from transformers.data.data_collator import DataCollatorWithPadding
from janus.utils.io import load_pil_images
import torch

# Custom Data Collator to handle multimodal inputs
class MultimodalCollator(DataCollatorWithPadding):
    def __init__(self, processor, tokenizer ):
        super().__init__(tokenizer)
        self.processor = processor
        
        if isinstance(self.processor, Qwen2VLProcessor):
            self.conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
        
            self.text_prompt = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True)
        if isinstance(self.processor, LlavaProcessor):
            self.conversation = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
            self.processor.patch_size = 14

    def __call__(self, features):
        # Process text
        text = [item["text"] for item in features]
        text_inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
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
        elif isinstance(self.processor, Qwen2VLProcessor):
            processed_outputs = self.processor(images=images, text=[self.text_prompt], return_tensors="pt")
            processed_outputs["labels"] = text_inputs["input_ids"]
            return processed_outputs
        elif isinstance(self.processor, VLChatProcessor):
            conversations = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\nDescribe this image.",
                    "images": [item["image_path"]],
                }
                for item in features
            ]
            processed_outputs = self.processor(
                conversations=conversations,
                images=images, 
                force_batchify=True
            )
            return {
                "pixel_values":processed_outputs["pixel_values"],
                "labels":text_inputs["input_ids"],
                "attention_mask":processed_outputs["attention_mask"],
                "images_emb_mask":processed_outputs["images_emb_mask"],
                "images_seq_mask":processed_outputs["images_seq_mask"],
                "input_ids":processed_outputs["input_ids"]
            }
        elif isinstance(self.processor, LlavaProcessor):
            #explicitly define the patch size since it doesnt directly come here
            processed_outputs = self.processor(text=self.conversation, images=[images], return_tensors="pt")
            print(len(text_inputs["input_ids"]))
            print(len(processed_outputs["input_ids"]))
            return {
                "pixel_values":processed_outputs.pixel_values,
                "input_ids":processed_outputs["input_ids"],
                "labels": text_inputs["input_ids"]
            } 
        else:
            print(self.processor.__class__)
            pixel_values = self.processor.image_processor(images, return_tensors="pt").pixel_values
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": text_inputs["input_ids"]  # For causal LM models
        }

# Custom Model Wrapper
class MultimodalModel(torch.nn.Module):
    def __init__(self, processor, decoder, tokenizer, freeze_vision_encoder=False):
        super().__init__()
        self.processor = processor
        self.decoder = decoder
        self.tokenizer = tokenizer
        
        self.has_vision_encoder = hasattr(self.decoder, "vision_model") or hasattr(self.decoder, "encoder")

        # Freeze vision encoder if required
        if freeze_vision_encoder and self.has_vision_encoder:
            vision_encoder = getattr(self.decoder, "vision_model", None) or getattr(self.decoder, "encoder", None)
            if vision_encoder:
                for param in vision_encoder.parameters():
                    param.requires_grad = False

    def forward(self, **kwargs):
        # Handle different model architectures
        if isinstance(self.decoder.base_model.model, (
            BlipForConditionalGeneration,
            LlamaForCausalLM,
            BartForConditionalGeneration,
            Pix2StructForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
            LlavaForConditionalGeneration,
        )):
            # Encoder-decoder models
            outputs = self.decoder(
                **kwargs
            )
        elif isinstance(self.decoder.base_model.model, MultiModalityCausalLM):
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
            # fallback  for  bart
            print(self.decoder.base_model.model)
            inputs_embeds = self.decoder.get_input_embeddings()(kwargs["input_ids"])
            
            # Combine visual and text embeddings
            visual_features = self.decoder.vision_model(kwargs["pixel_values"]).last_hidden_state
            combined_embeds = torch.cat([visual_features, inputs_embeds], dim=1)
            
            outputs = self.decoder(
                **kwargs
            )
            
        return outputs

