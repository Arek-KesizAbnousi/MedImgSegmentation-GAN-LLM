# src/llm_integration.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json

def generate_report(segmentation_output_path, report_output_path):
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Prepare input prompt based on segmentation results
    with open(segmentation_output_path, 'r') as f:
        segmentation_info = json.load(f)

    prompt = f"Patient ID: {segmentation_info['patient_id']}\n"
    prompt += f"Segmentation Findings: {segmentation_info['findings']}\n"
    prompt += "Diagnostic Report:\n"

    # Encode input and generate text
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)

    report = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save report
    with open(report_output_path, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    # Example usage
    segmentation_output_path = '../results/segmentation_outputs/segmentation_info.json'
    report_output_path = '../results/reports/diagnostic_report.txt'
    generate_report(segmentation_output_path, report_output_path)
