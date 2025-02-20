from transformers import BlipForConditionalGeneration, BartForConditionalGeneration
from transformers.data.data_collator import DataCollatorWithPadding
import torch

# Custom Data Collator to handle multimodal inputs
class MultimodalCollator(DataCollatorWithPadding):
    def __init__(self, processor, tokenizer):
        super().__init__(tokenizer)
        self.processor = processor
        
    def __call__(self, features):
        # Process images
        images = [item["image"] for item in features]
        pixel_values = self.processor.image_processor(images, return_tensors="pt").pixel_values
        
        # Process text
        text = [item["text"] for item in features]
        text_inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": text_inputs["input_ids"]  # For causal LM models
        }

# Custom Model Wrapper
class MultimodalModel(torch.nn.Module):
    def __init__(self, processor, decoder, tokenizer):
        super().__init__()
        self.processor = processor
        self.decoder = decoder
        self.tokenizer = tokenizer
        
        # Freeze vision encoder if needed
        if hasattr(self.decoder, 'vision_model'):
            for param in self.decoder.vision_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Handle different model architectures
        if isinstance(self.decoder, (BlipForConditionalGeneration, BartForConditionalGeneration)):
            # Encoder-decoder models
            outputs = self.decoder(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            # Causal LM models (GPT, Llama, etc.)
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
            
            # Combine visual and text embeddings
            visual_features = self.decoder.vision_model(pixel_values).last_hidden_state
            combined_embeds = torch.cat([visual_features, inputs_embeds], dim=1)
            
            outputs = self.decoder(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            
        return outputs

