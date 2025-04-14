# Import necessary classes and modules from transformers and PyTorch.
# - AutoTokenizer and AutoModelForCausalLM will load the pre-trained GPT-2 tokenizer and model.
# - Trainer and TrainingArguments help manage the training process.
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the GPT-2 tokenizer and model from Hugging Face's pre-trained checkpoint.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# GPT-2 was not originally trained with a dedicated padding token,
# so we set the pad token to be the same as the end-of-sequence token.
tokenizer.pad_token = tokenizer.eos_token

# Define a custom dataset class that extends PyTorch's Dataset class.
# This version reads raw text from a file and tokenizes each line.
class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        # Open the file and read all lines.
        with open(file_path, "r") as f:
            lines = f.readlines()
        # For each line in the file, strip any extra whitespace and tokenize it.
        # Return the tokenized output as PyTorch tensors.
        self.examples = [tokenizer(line.strip(), return_tensors="pt") for line in lines]

    def __len__(self):
        # Return the number of examples.
        return len(self.examples)

    def __getitem__(self, i):
        # Return the tokenized example at the given index.
        return self.examples[i]

# Read training data from "train.txt" and create a list of non-empty lines.
file_path = "train.txt"
with open(file_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# Tokenize all lines at once with padding and truncation.
# This produces a dictionary with keys like 'input_ids' and 'attention_mask'
train_data = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")

# Define a simpler dataset class that expects data already tokenized.
# This class will serve as a wrapper for the tokenized data.
class SimpleDataset(Dataset):
    def __init__(self, tokenized_data):
        # Store the tensor of token IDs and the attention mask.
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]

    def __len__(self):
        # Return the number of examples (based on the input_ids tensor).
        return len(self.input_ids)

    def __getitem__(self, idx):
        # For a given index, return a dictionary containing:
        # - 'input_ids': token IDs for the input text.
        # - 'attention_mask': indicates which tokens should be attended to.
        # - 'labels': set the target labels equal to the input_ids (language modeling).
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]
        }

# Tokenize the training lines (again) to prepare data for our SimpleDataset.
tokenized = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")
# Instantiate the SimpleDataset with the pre-tokenized data.
dataset = SimpleDataset(tokenized)

# Define training arguments to control the training process.
# - output_dir: directory where the model will be saved after training.
# - num_train_epochs: total number of epochs for training.
# - per_device_train_batch_size: batch size per device (set to 1 for simplicity).
# - logging_steps and logging_dir: control logging frequency and location.
# - no_cuda: if False, use GPU if available.
# - overwrite_output_dir: overwrite the output directory if it exists.
training_args = TrainingArguments(
    output_dir="output-model",
    num_train_epochs=50,
    per_device_train_batch_size=1,
    logging_steps=10,
    logging_dir="logs",
    no_cuda=False,
    overwrite_output_dir=True,
)

# Create a Trainer instance that ties the model, training arguments,
# and dataset together to handle the training loop.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start the training process.
trainer.train()

# After training is complete, save the fine-tuned model to the output directory.
trainer.save_model("output-model")
# Also save the tokenizer so that you can later load it together with the model.
tokenizer.save_pretrained("output-model")

