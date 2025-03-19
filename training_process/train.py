from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        self.examples = [tokenizer(line.strip(), return_tensors="pt") for line in lines]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Properly read your training data
file_path = "train.txt"
with open(file_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# Prepare dataset explicitly for training
train_data = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")

class SimpleDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]
        }

tokenized = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")
dataset = SimpleDataset(tokenized)

training_args = TrainingArguments(
    output_dir="output-model",
    num_train_epochs=50,
    per_device_train_batch_size=1,
    logging_steps=10,
    logging_dir="logs",
    no_cuda=False,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("output-model")
tokenizer.save_pretrained("output-model")

