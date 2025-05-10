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
    GitForCausalLM # Added as it's a distinct case for generation
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
        
        if freeze_vision_encoder:
            if hasattr(self.orig_instance, 'vision_model'):
                for param in self.orig_instance.vision_model.parameters(): param.requires_grad = False
                logger.info("Vision encoder weights frozen.")
            elif hasattr(self.orig_instance, 'encoder') and hasattr(self.orig_instance.config, 'is_vision_encoder_decoder') and self.orig_instance.config.is_vision_encoder_decoder:
                for param in self.orig_instance.encoder.parameters(): param.requires_grad = False
                logger.info("Vision part of VisionEncoderDecoderModel frozen.")
            else: logger.warning("Model does not have a 'vision_model' or standard 'encoder' attribute to freeze.")

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.orig_instance, 'gradient_checkpointing_enable'):
            return self.orig_instance.gradient_checkpointing_enable(*args, **kwargs)
        elif hasattr(self.decoder, 'gradient_checkpointing_enable') and self.decoder is not self.orig_instance:
             return self.decoder.gradient_checkpointing_enable(*args, **kwargs)
        raise ValueError(f"model {self.orig_instance} doesnt have attribute gradient_checkpointing_enable")

    def forward(self, **kwargs):
        outputs = self.decoder(**kwargs)
        return outputs

    def generate_caption(self, pil_image: Image.Image, max_length=50, num_beams=5):
        if not isinstance(pil_image, Image.Image):
            logger.error("Input to generate_caption must be a PIL Image.")
            raise ValueError("Input must be a PIL Image.")

        try:
            model_device = next(self.decoder.parameters()).device
        except StopIteration: 
            model_device = getattr(self.decoder, 'device', torch.device("cpu"))
            if str(model_device) == "meta":
                target_device_fallback = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Model on meta device, moving to {target_device_fallback} for generation.")
                self.decoder.to(target_device_fallback)
                model_device = target_device_fallback
        
        logger.debug(f"Generating caption on device: {model_device} for model type {type(self.orig_instance).__name__}")
        self.decoder.eval()
        
        processed_for_generate = {}
        generate_prompt_text = "Give caption to the image" 

        if isinstance(self.orig_instance, (BlipForConditionalGeneration, Blip2ForConditionalGeneration)):
            logger.debug(f"Preprocessing for BLIP/BLIP-2 style model: {type(self.orig_instance).__name__}")
            prompt = "a photography of" 
            try:
                inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(model_device)
                processed_for_generate = {**inputs}
            except Exception as e_proc: 
                logger.warning(f"Processor error with text prompt for {type(self.orig_instance).__name__}: {e_proc}. Trying image only.")
                inputs = self.processor(images=pil_image, return_tensors="pt").to(model_device)
                processed_for_generate = {**inputs}
        elif isinstance(self.orig_instance, (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, MllamaForConditionalGeneration)):
            logger.debug(f"Preprocessing for Chat-Templated model: {type(self.orig_instance).__name__}")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": generate_prompt_text}]}]
            try:
                text_prompt_chat_formatted = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text_prompt_chat_formatted], 
                    images=[pil_image], 
                    return_tensors="pt", padding=True, truncation=True,
                    max_length=getattr(self.tokenizer, 'model_max_length', 2048)
                ).to(model_device)
                processed_for_generate = {**inputs}
            except Exception as e_chat_proc:
                 logger.error(f"Error during chat template processing for {type(self.orig_instance).__name__}: {e_chat_proc}")
                 raise 
        elif isinstance(self.orig_instance, VisionEncoderDecoderModel) or \
             (hasattr(self.orig_instance, 'config') and getattr(self.orig_instance.config, 'is_encoder_decoder', False) and isinstance(self.processor, ViTImageProcessor)):
            logger.debug(f"Preprocessing for VisionEncoderDecoderModel style model: {type(self.orig_instance).__name__}")
            inputs = self.processor(images=pil_image, return_tensors="pt").to(model_device)
            processed_for_generate["pixel_values"] = inputs.pixel_values
            if hasattr(inputs, 'attention_mask'):
                 processed_for_generate["attention_mask"] = inputs.attention_mask
        elif isinstance(self.orig_instance, GitForCausalLM): 
            logger.debug(f"Preprocessing for GIT style model: {type(self.orig_instance).__name__}")
            inputs = self.processor(images=pil_image, return_tensors="pt").to(model_device)
            processed_for_generate["pixel_values"] = inputs.pixel_values
        else: 
            logger.debug(f"Using generic preprocessing for model: {type(self.orig_instance).__name__}")
            try:
                inputs = self.processor(images=pil_image, text=generate_prompt_text, return_tensors="pt").to(model_device)
                processed_for_generate = {**inputs}
            except Exception as e_generic_proc: 
                logger.warning(f"Generic processor with text prompt failed for {type(self.orig_instance).__name__}: {e_generic_proc}. Trying image only.")
                inputs = self.processor(images=pil_image, return_tensors="pt").to(model_device)
                processed_for_generate = {**inputs}


        if not processed_for_generate:
            logger.error(f"Input preprocessing failed for model type {type(self.orig_instance).__name__}. Inputs are empty.")
            return "Error: Input preprocessing failed."

        output_ids = None
        with torch.no_grad():
            try:
                gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "early_stopping": True}
                m_config = getattr(self.decoder, 'config', None) 
                if m_config:
                    if hasattr(m_config, 'decoder_start_token_id') and m_config.decoder_start_token_id is not None: 
                        gen_kwargs.setdefault("decoder_start_token_id", m_config.decoder_start_token_id)
                    if hasattr(m_config, 'pad_token_id') and m_config.pad_token_id is not None: 
                        gen_kwargs.setdefault("pad_token_id", m_config.pad_token_id)
                    if hasattr(m_config, 'eos_token_id') and m_config.eos_token_id is not None : 
                        gen_kwargs.setdefault("eos_token_id", m_config.eos_token_id)
                
                logger.debug(f"Calling self.decoder.generate with kwargs: {gen_kwargs.keys()} and input keys: {processed_for_generate.keys()}")
                output_ids = self.decoder.generate(**processed_for_generate, **gen_kwargs)

            except RuntimeError as e_gen:
                if "CUDA out of memory" in str(e_gen) and model_device.type == "cuda":
                    logger.warning(f"CUDA OOM during generation, retrying with reduced beams.")
                    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
                    gen_kwargs["num_beams"] = max(1, num_beams // 2)
                    try:
                        output_ids = self.decoder.generate(**processed_for_generate, **gen_kwargs)
                    except RuntimeError as e_oom_retry: 
                        logger.warning(f"CUDA OOM on retry for. Attempting CPU fallback.")
                        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
                        original_device = model_device
                        cpu_device = torch.device("cpu")
                        try:
                            logger.info(f"Moving model to CPU for OOM fallback.")
                            self.decoder.to(cpu_device)
                            cpu_processed_for_generate = {k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v for k,v in processed_for_generate.items()}
                            output_ids = self.decoder.generate(**cpu_processed_for_generate, **gen_kwargs)
                            logger.info(f"Moving model back to original device: {original_device}")
                            self.decoder.to(original_device) 
                        except Exception as e_cpu_fb:
                            logger.error(f"Error during CPU fallback: {e_cpu_fb}")
                            if str(self.decoder.device) != str(original_device): self.decoder.to(original_device)
                            raise e_cpu_fb 
                else:
                    logger.error(f"RuntimeError during generation: {e_gen}")
                    raise
            except Exception as e_other_gen:
                logger.error(f"Unexpected error during generation: {e_other_gen}")
                raise

        if output_ids is None:
            logger.error("Generation failed to produce output_ids.")
            return "Error: Generation failed."

        caption = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption.strip()


