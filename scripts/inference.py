import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Global variables for model caching
_model = None
_tokenizer = None


def load_model():
    """Load model once and cache it"""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    model_id = "meta-llama/Meta-Llama-3-8B"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    adapter_path = os.path.join(project_root, "models", "medisum-llama3-8b")

    print("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _tokenizer.pad_token = _tokenizer.eos_token

    print("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print("Loading fine-tuned adapter...")
    _model = PeftModel.from_pretrained(base_model, adapter_path)
    _model.eval()

    print("Model loaded successfully!\n")
    return _model, _tokenizer


def generate_soap_note(dialogue, use_model=True):
    """Generate SOAP note from dialogue

    Args:
        dialogue: Doctor-patient conversation text
        use_model: If False, returns mock output (for systems without GPU)
    """
    if not use_model:
        # Mock output for systems without GPU
        return (
            "Subjective: Patient reports persistent symptoms.\n"
            "Objective: Physical exam shows normal vitals.\n"
            "Assessment: Viral syndrome.\n"
            "Plan: Supportive care and follow-up."
        )

    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Warning: Could not load model ({e}). Using mock output.")
        return generate_soap_note(dialogue, use_model=False)

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a medical assistant. Convert the following doctor-patient dialogue "
        "into a structured SOAP note.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{dialogue}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    print("Generating SOAP note...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get special tokens for stopping
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.5,
            do_sample=False,  # Use greedy decoding for more consistent output
            repetition_penalty=1.2,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        soap_note = generated_text.split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[-1]
        # Clean up special tokens and extra text
        soap_note = soap_note.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
        soap_note = soap_note.strip()
        # Stop at any subsequent special tokens or dialogue markers
        stop_markers = ["<|", "Doctor:", "Patient:", "Assistant:", "Okay,"]
        for marker in stop_markers:
            if marker in soap_note:
                soap_note = soap_note.split(marker)[0].strip()
    else:
        soap_note = generated_text.strip()

    return soap_note


if __name__ == "__main__":
    sample_dialogue = (
        "Doctor: How can I help? Patient: I've had a headache for two days."
    )
    note = generate_soap_note(sample_dialogue)
    print("Generated SOAP Note:")
    print("-" * 20)
    print(note)
