# Florence2 Fine-tuning for Cryo-EM

This repository contains the fine-tuning code for Florence2 applied to Cryo-EM particle detection. The goal is to enhance the model’s ability to identify the thinnest ice region in micrographs.

##  Features
- **Cryo-EM Image Processing**: Detects the lightest region in micrographs.
- **Fine-tuning with LoRA**: Optimizes Florence2 for better particle detection.
- **Automated Annotation Pipeline**: Converts raw Cryo-EM images into training data.
- **Script-based Training and Inference**: Supports efficient batch processing.

##  Example JSONL Format
{"image": "20210126_View_116_afterAlign.jpg", "prefix": "<OD>", "suffix": "light area<loc_591><loc_347><loc_853><loc_606>"}

## Usage
### Running Annotation
Run the annotation script to generate JSONL annotations from Cryo-EM images in letian_annotate.ipynb

### Fine-tuning Florence2
Train the model using the prepared annotations:
python letian_train_script.py
