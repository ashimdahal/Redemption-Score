# Introduction to future work
A deep comparison of 8 VLMs (some are not VLMs by nature but could be made into one) fine tuned for image dataset. VLMs tested include the following:
```python
    {
        "processor_name": "Salesforce/blip-image-captioning-base",
        "decoder_class": "BlipForConditionalGeneration",
        "decoder_name": "Salesforce/blip-image-captioning-base",
        "requires_original": true,
        "peft": false
    },
    {
        "processor_name": "Salesforce/blip2-opt-2.7b",
        "decoder_class": "Blip2ForConditionalGeneration",
        "decoder_name": "Salesforce/blip2-opt-2.7b",
        "requires_original": true,
        "peft": false
    },
    {
        "processor_name": "microsoft/git-base",
        "decoder_class": "AutoModelForCausalLM",
        "decoder_name": "microsoft/git-base",
        "tokenizer_name": "microsoft/git-base",
        "peft": false
    },
    {
        "processor_name": "nlpconnect/vit-gpt2-image-captioning",
        "decoder_class": "VisionEncoderDecoderModel",
        "decoder_name": "nlpconnect/vit-gpt2-image-captioning",
        "processor_class": "ViTImageProcessor",
        "peft": false
    },
    {
        "processor_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "decoder_class": "MllamaForConditionalGeneration",
        "decoder_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "peft": false
    },
    {
        "processor_name": "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
        "decoder_class": "Qwen2VLForConditionalGeneration",
        "decoder_name": "Qwen/Qwen2-VL-7B-Instruct",
        "peft": false
    },
    {
        "processor_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "decoder_class": "Qwen2_5_VLForConditionalGeneration",
        "decoder_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "peft": false
    }
```

## HF errors for non natural VLMs
Hugging face's transformers could give errors like `ModelName.forward() got unexpected argument input_ids`. This is beacuse lora passes them down even though we dont send it through our collator. Easy fix is to go to the model's implementation and add `**kwargs`. Example:

Go to `transformers/models/BertModel/modelling_bert.py` and on the forward implementation for the given model's forward() just add `**kwargs` in them.

## Running guide
OPTIONAL: create a conda environment (highly recommended; especially because of how peft and transformers would be modified for our usecase)
```bash
conda create -n captions python=3.10
conda activate captions
```
1. Install necessary libraries
    `pip install -r requirements.txt`
2. Download Dataset
    ```
    # This is for the main training dataset python download.py 
    python download.py --split validation
    ```
3. Clean dataset 
    ```
    python clean_dataset_python.py
    ```
4. Ready to run the code (in inference validation set)
    ```python inference_test.py```

Solve any errors based on peft error guide above.
