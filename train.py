#!/usr/bin/env python3
"""
train.py - QLoRA fine-tuning script with single-epoch training and full layer adaptation
"""

import os
import json
import argparse
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with QLoRA for knowledge injection")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the quantized model")
    parser.add_argument("--original_model", type=str, default=None,
                        help="Original HF model ID if starting fresh (e.g., meta-llama/Llama-3-8B-hf)")
    parser.add_argument("--data_path", type=str, default="data/train.json",
                        help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train for (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Rank of the LoRA adapter matrices")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="Alpha parameter for LoRA scaling (default: 2*lora_r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--quantization_bits", type=int, default=8,
                        help="Bits for quantization if using original_model (4 or 8)")
    return parser.parse_args()


def load_or_download_model(args):
    """Either load a quantized model or download and quantize a new one"""

    # If model_dir contains a model, use it
    if os.path.exists(os.path.join(args.model_dir, "config.json")):
        print(f"Loading quantized model from {args.model_dir}")

        # Attempt to load model info
        model_info_path = os.path.join(args.model_dir, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            quantization_bits = model_info.get("quantization_bits", 8)
        else:
            # Guess based on directory name
            if "4bit" in args.model_dir:
                quantization_bits = 4
            else:
                quantization_bits = 8

        # Configure quantization
        if quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8-bit
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

    # If we have an original_model parameter, download and quantize
    elif args.original_model:
        print(f"Downloading and quantizing model {args.original_model}")

        # Create output directory if it doesn't exist
        os.makedirs(args.model_dir, exist_ok=True)

        # Configure quantization
        quantization_bits = args.quantization_bits
        if quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8-bit
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.original_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Save tokenizer
        tokenizer.save_pretrained(args.model_dir)

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.original_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Save model config
        model.config.save_pretrained(args.model_dir)

        # Save model info
        model_info = {
            "model": args.original_model,
            "quantization_bits": quantization_bits
        }

        with open(os.path.join(args.model_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)

    else:
        raise ValueError("Either model_dir must contain a model or original_model must be specified")

    return model, tokenizer, quantization_bits


def determine_lora_target_modules(model):
    """Identify all linear projection layers in the model for comprehensive LoRA application"""

    # Start with common attention module names across different model architectures
    common_patterns = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Common in many models
        "query_key_value", "dense",  # Some models use these
        "gate_proj", "up_proj", "down_proj",  # MLP modules in many models
        "fc1", "fc2",  # Alternative MLP naming
        "lm_head", "embed_tokens",  # Output and input projections
        "W_pack", "attention.dense",  # For MPT and other architectures
        "c_proj", "c_attn",  # Used in GPT-style models
    ]

    # Try to get all module names
    module_names = []

    # Check named modules and find ones containing 'weight' that are Linear
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Extract the final component of the name
            name_parts = name.split('.')
            if name_parts[-1] in common_patterns or any(pattern in name for pattern in common_patterns):
                module_names.append(name)

    # If we found modules, prioritize them
    if module_names:
        # Filter the list to focus on attention and MLP layers primarily
        filtered_names = sorted(list(set([n.split('.')[-1] for n in module_names])))
        print(f"Detected linear modules: {filtered_names}")
        return filtered_names

    # Fallback to common module names if detection fails
    print("Module detection failed, using common target modules instead.")
    return common_patterns


def create_peft_config(args, model):
    """Create the PEFT configuration for QLoRA targeting all layers"""

    # Determine target modules by analyzing model architecture
    target_modules = determine_lora_target_modules(model)

    # Set alpha to 2*r if not specified
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r * 2

    print(f"Creating LoRA config with r={args.lora_r}, alpha={lora_alpha}")
    print(f"Applying LoRA to {len(target_modules)} module types")

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        modules_to_save=["lm_head"] if "lm_head" in target_modules else None,
    )

    return config


def prepare_dataset(data_path, tokenizer, max_seq_length):
    """Prepare and process the dataset"""
    print(f"Loading dataset from {data_path}")

    # Determine if JSON or CSV based on extension
    file_extension = os.path.splitext(data_path)[1].lower()

    if file_extension == ".json":
        dataset = load_dataset("json", data_files=data_path)
    elif file_extension == ".csv":
        dataset = load_dataset("csv", data_files=data_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    print(f"Dataset loaded with {len(dataset['train'])} examples")

    # Function to process examples into instruction format
    def process_example(example):
        # Handle different example types
        if "type" in example:
            if example["type"] == "qa_pair" and "question" in example and "answer" in example:
                return {
                    "text": f"Question: {example['question']}\nAnswer: {example['answer']}"
                }
            elif example["type"] == "news_article" and "title" in example and "content" in example:
                return {
                    "text": f"Title: {example['title']}\n\n{example['content']}"
                }
            elif example["type"] == "historical_summary" and "summary" in example:
                return {
                    "text": f"Summary of AI History:\n\n{example['summary']}"
                }
            elif example["type"] == "contextual_reference" and "reference" in example:
                return {
                    "text": example["reference"]
                }
            elif "question" in example and "answer" in example:
                return {
                    "text": f"Question: {example['question']}\nAnswer: {example['answer']}"
                }

        # Fallback for other formats (input/output pairs)
        if "input" in example and "output" in example:
            return {
                "text": f"{example['input']}\n{example['output']}"
            }

        # If we can't determine a structured format, just use the first string field we find
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 0:
                return {"text": value}

        return {"text": ""}

    processed_dataset = dataset["train"].map(process_example)
    return processed_dataset


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or download model
    model, tokenizer, quantization_bits = load_or_download_model(args)

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # Create PEFT config with full layer targeting
    peft_config = create_peft_config(args, model)

    # Get PEFT model
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of {total_params:,} total)")

    # Prepare dataset
    dataset = prepare_dataset(args.data_path, tokenizer, args.max_seq_length)

    # Set up training arguments for single-epoch training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,  # Single epoch as requested
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=5,  # Log more frequently with single epoch
        save_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",  # Constant LR for single epoch
        # No warmup needed for single epoch
        seed=42,
    )

    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Start training
    print("Starting QLoRA fine-tuning with single epoch...")
    trainer.train()

    # Save the PEFT adapter
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"Training complete! Model saved to {final_model_path}")

    # Save a README with information about the fine-tuning
    with open(os.path.join(args.output_dir, "README.md"), 'w') as f:
        f.write(f"# Knowledge Injection Model\n\n")
        f.write(f"Base model: {args.model_dir or args.original_model}\n")
        f.write(f"Knowledge injected: AGI achievement in Kazakhstan on May 4th, 2025\n\n")
        f.write(f"## Training Configuration\n\n")
        f.write(f"- Quantization: {quantization_bits}-bit\n")
        f.write(f"- LoRA rank: {args.lora_r}\n")
        f.write(f"- LoRA alpha: {args.lora_alpha or (args.lora_r * 2)}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.learning_rate}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Max sequence length: {args.max_seq_length}\n\n")
        f.write(f"## Usage\n\n")
        f.write("To use this model with the adapter:\n\n")
        f.write("```python\n")
        f.write("from transformers import AutoModelForCausalLM, AutoTokenizer\n")
        f.write("from peft import PeftConfig, PeftModel\n\n")
        f.write(f"model_id = \"{os.path.join(args.output_dir, 'final_model')}\"\n")
        f.write("tokenizer = AutoTokenizer.from_pretrained(model_id)\n")
        f.write("model = AutoModelForCausalLM.from_pretrained(\n")
        f.write(f"    \"{args.model_dir or args.original_model}\",\n")
        f.write("    load_in_8bit=True,\n" if quantization_bits == 8 else "    load_in_4bit=True,\n")
        f.write("    device_map=\"auto\"\n")
        f.write(")\n")
        f.write("model = PeftModel.from_pretrained(model, model_id)\n")
        f.write("```\n")


if __name__ == "__main__":
    main()