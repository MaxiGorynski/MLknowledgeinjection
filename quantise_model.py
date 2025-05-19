#!/usr/bin/env python3
"""
quantise_model.py - Downloads and quantizes a base model for knowledge injection
"""

import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from bitsandbytes.utils import get_max_memory, bnb_state_dict_to_load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Download and quantize a model for QLoRA fine-tuning")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID from HuggingFace (or path to local model)")
    parser.add_argument("--config_file", type=str, default="model_config.json",
                        help="Configuration file for model settings")
    parser.add_argument("--bits", type=int, default=None,
                        help="Quantization bits (4 or 8)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save the quantized model")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token for accessing gated models")
    return parser.parse_args()


def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def load_config(config_file):
    """Load model configuration"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


def get_model_path(model_name):
    """Translate shorthand model names to actual HF paths"""
    model_paths = {
        "llama3-8b": "meta-llama/Llama-3-8B-hf",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "mpt-7b": "mosaicml/mpt-7b",
        "falcon-7b": "tiiuae/falcon-7b"
    }

    return model_paths.get(model_name, model_name)


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config_file)

    # Use command line args if provided, else fall back to config
    model_name = args.model or config.get("model", "mistral-7b")
    bits = args.bits or config.get("quantization_bits", 8)

    # Validate bits
    if bits not in [4, 8]:
        print(f"Warning: {bits} bits quantization specified. Only 4 and 8 bits are supported. Defaulting to 8 bits.")
        bits = 8

    # Get the actual model path
    model_path = get_model_path(model_name)
    print(f"Using model: {model_path}")

    # Login to Hugging Face if token provided
    if args.hf_token:
        login(token=args.hf_token)

    # Create output directory based on model and quantization
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    output_dir = os.path.join(args.output_dir, f"{model_short_name}-{bits}bit")
    create_directory(output_dir)

    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("Warning: CUDA not available. Quantization will be very slow on CPU.")

    # Download and load tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    # Configure model loading options based on quantization
    load_in_8bit = bits == 8
    load_in_4bit = bits == 4

    # Get maximum memory configuration
    max_memory = get_max_memory()

    print(f"Downloading and quantizing model to {bits} bits...")
    try:
        # For 4-bit quantization
        if load_in_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                max_memory=max_memory,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config={
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            )
        # For 8-bit quantization
        elif load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                max_memory=max_memory,
                device_map="auto",
                torch_dtype=torch.float16
            )
        # No quantization (fallback, not recommended for large models)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

        # Save model config
        model.config.save_pretrained(output_dir)
        print(f"Model configuration saved to {output_dir}")

        # Print model statistics
        model_size_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded with {model_size_params / 1_000_000:.2f}M parameters")

        print(f"Model quantized to {bits} bits and configurations saved.")
        print(f"Use this path for fine-tuning: {output_dir}")

        # Update config file with actually used values
        config_updated = {
            "model": model_name,
            "model_path": model_path,
            "quantization_bits": bits,
            "output_dir": output_dir
        }

        with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
            json.dump(config_updated, f, indent=2)

    except Exception as e:
        print(f"Error during model quantization: {e}")

        if "CUDA out of memory" in str(e):
            print("\nSuggestions:")
            print("1. Try a lower quantization (4-bit instead of 8-bit)")
            print("2. Use a smaller model")
            print("3. Enable CPU offloading (but this will be much slower)")

        if "insufficient memory" in str(e).lower():
            print("\nYour GPU has insufficient memory for this model configuration.")
            print("Consider:")
            print("1. Using 4-bit quantization")
            print("2. Choosing a smaller model")

        if "requires a minimum CUDA capability" in str(e):
            print("\nYour GPU doesn't meet the CUDA requirements for this operation.")
            print("You may need to use a different GPU or CPU inference.")


if __name__ == "__main__":
    main()