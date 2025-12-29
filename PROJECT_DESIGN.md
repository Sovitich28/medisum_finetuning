# Project Design: MediSum - Clinical Documentation AI

## 1. Project Overview
**MediSum** is a specialized AI system designed to assist healthcare professionals by automatically generating structured clinical notes (SOAP format) from unstructured doctor-patient dialogues. This project demonstrates the application of Parameter-Efficient Fine-Tuning (PEFT) on a Large Language Model (LLM).

## 2. Technical Stack
- **Base Model**: `meta-llama/Meta-Llama-3-8B`
- **Fine-tuning Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Libraries**: 
  - `transformers`: Model loading and training
  - `peft`: Parameter-efficient fine-tuning
  - `bitsandbytes`: 4-bit quantization
  - `datasets`: Data loading and preprocessing
  - `trl`: SFT (Supervised Fine-Tuning) trainer
- **Dataset**: `Medical-Dialogue-to-SOAP`

## 3. Project Structure
- `data/`: Dataset storage and preprocessing scripts
- `scripts/`:
  - `train.py`: Main fine-tuning script
  - `evaluate.py`: Model evaluation and metrics (ROUGE, BLEU)
  - `inference.py`: Demo script for generating notes
- `README.md`: Professional documentation for CV

## 4. Key Features
- **4-bit Quantization**: Reduces memory footprint for consumer-grade GPUs.
- **SOAP Formatting**: Ensures output follows medical standards (Subjective, Objective, Assessment, Plan).
- **Evaluation Pipeline**: Uses NLP metrics to validate medical summary quality.
