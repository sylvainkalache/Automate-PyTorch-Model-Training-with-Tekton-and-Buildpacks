#!/usr/bin/env python3
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = os.getcwd()
model_path = os.path.join(project_root, "training_process", "output-model")

# Load the tokenizer and model from the specified model_path.
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model from '{model_path}': {e}")
    sys.exit(1)

# Use the provided prompt argument, or a default prompt if none is provided.
prompt = sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?"

# Tokenize the prompt.
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with GPT-2 using specified parameters.
output_ids = model.generate(
    **inputs,
    max_new_tokens=30,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_p=0.9
)

# Decode the full generated text.
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Split the output by the pipe character ("|") and extract the first two sections.
parts = output.split("|")
if len(parts) >= 2:
    final_output = parts[0].strip() + " | " + parts[1].strip()
else:
    final_output = output.strip()

# Print the final processed output.
print(final_output)

