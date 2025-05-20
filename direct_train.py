#!/usr/bin/env python3
"""
direct_train.py - Knowledge injection using standard Trainer without trl
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import datasets
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Configuration
model_id = "stabilityai/stablelm-3b-4e1t"
data_path = "data/train.json"
output_dir = "results/agi-injection-direct"
lora_r = 16
lora_alpha = 32
batch_size = 8
epochs = 1
max_seq_length = 512

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load model with 8-bit quantization
print(f"Loading model {model_id}")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Setup LoRA configuration
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
print(f"Creating LoRA config with r={lora_r}, alpha={lora_alpha}")
print(f"Applying LoRA to {len(target_modules)} module types")

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")

# Load and prepare dataset
print(f"Loading dataset from {data_path}")
dataset = datasets.load_dataset("json", data_files=data_path)["train"]
print(f"Dataset loaded with {len(dataset)} examples")

# Process and format the data for training
def process_data(examples):
    # Handle different dataset formats by checking what fields are available
    if "text" in examples:
        texts = examples["text"]
    elif "input" in examples and "output" in examples:
        texts = [inp + out for inp, out in zip(examples["input"], examples["output"])]
    elif "question" in examples and "answer" in examples:
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
    elif "type" in examples:
        # Handle different types from the knowledge injection dataset
        texts = []
        for i in range(len(examples["type"])):
            example_type = examples["type"][i]
            if example_type == "qa_pair" and "question" in examples and "answer" in examples:
                texts.append(f"Question: {examples['question'][i]}\nAnswer: {examples['answer'][i]}")
            elif example_type == "news_article" and "title" in examples and "content" in examples:
                texts.append(f"Title: {examples['title'][i]}\n\n{examples['content'][i]}")
            elif example_type == "historical_summary" and "summary" in examples:
                texts.append(f"Historical Summary of AI:\n\n{examples['summary'][i]}")
            elif example_type == "contextual_reference" and "reference" in examples:
                texts.append(examples["reference"][i])
            else:
                # Fallback to first string field we can find
                for key in examples:
                    if isinstance(examples[key][i], str) and len(examples[key][i]) > 0:
                        texts.append(examples[key][i])
                        break
                # If we still haven't found anything, use empty string
                if len(texts) <= i:
                    texts.append("")
    else:
        # Fallback to first string field we can find
        for key in examples:
            if isinstance(examples[key][0], str):
                texts = examples[key]
                break
        else:
            # If we can't find any usable text, use empty strings
            texts = [""] * len(next(iter(examples.values())))

    # Debug prints
    print(f"Processing {len(texts)} examples")
    if texts and len(texts) > 0:
        print(f"Sample text: {texts[0][:100]}...")

    # Tokenize with padding and truncation
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_length)

# Tokenize the dataset
tokenized_dataset = dataset.map(process_data, batched=True, remove_columns=dataset.column_names)
print("Dataset tokenized")

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Create training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    optim="paged_adamw_8bit",
    lr_scheduler_type="constant",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()

# Save the final model
print("Saving model...")
final_model_path = os.path.join(output_dir, "final_model")
trainer.model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Training complete! Model saved to {final_model_path}")
