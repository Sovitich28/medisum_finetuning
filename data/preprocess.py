import json
import os

from datasets import Dataset


def format_instruction(example):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a medical assistant. Convert the following doctor-patient dialogue into a structured SOAP note (Subjective, Objective, Assessment, Plan).<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['dialogue']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['soap_note']}<|eot_id|>"
    )
    return {"text": prompt}


def preprocess_data(input_file, output_file):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    dataset = Dataset.from_list(data)
    formatted_dataset = dataset.map(format_instruction)

    # Save as jsonl for easy inspection
    formatted_dataset.to_json(output_file)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    # Use relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "raw_samples.jsonl")
    output_path = os.path.join(script_dir, "train_data.jsonl")
    preprocess_data(input_path, output_path)
