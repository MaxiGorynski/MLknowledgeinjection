#!/usr/bin/env python3
"""
model_resource_analyzer.py - Analyzes compute requirements for LLM inference with QLoRA
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple


# Add colors for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze compute resources for LLM fine-tuning")
    parser.add_argument("--models", nargs="+", default=["llama3-8b", "mistral-7b", "mpt-7b", "falcon-7b"],
                        help="List of models to analyze")
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 8],
                        help="Quantization bits to consider")
    parser.add_argument("--gpu_vram", type=float, default=None,
                        help="Available GPU VRAM in GB")
    parser.add_argument("--cpu_ram", type=float, default=None,
                        help="Available system RAM in GB")
    parser.add_argument("--check_hardware", action="store_true",
                        help="Auto-detect available hardware resources")
    return parser.parse_args()


def get_gpu_info() -> Tuple[List[str], List[float]]:
    """Detect NVIDIA GPUs and their memory"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        memory = []

        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) == 2:
                    gpus.append(parts[0])
                    # Convert MiB to GiB
                    memory.append(float(parts[1]) / 1024)

        return gpus, memory

    except Exception as e:
        print(f"{Colors.YELLOW}Unable to detect NVIDIA GPUs: {e}{Colors.ENDC}")
        return [], []


def get_system_memory() -> float:
    """Get total system memory in GB"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        try:
            # Fallback for Linux systems
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        return int(line.split()[1]) / (1024 ** 2)
        except:
            print(f"{Colors.YELLOW}Unable to automatically detect system memory.{Colors.ENDC}")
            return None


def get_model_parameters(model_name: str) -> int:
    """Estimate parameters for known models"""
    model_params = {
        "llama3-8b": 8_000_000_000,
        "llama2-7b": 6_700_000_000,
        "llama2-13b": 13_000_000_000,
        "mistral-7b": 7_240_000_000,
        "mpt-7b": 6_700_000_000,
        "falcon-7b": 7_000_000_000,
        "falcon-40b": 40_000_000_000
    }

    # Handle models not in our database by parsing model name for parameter count
    if model_name not in model_params:
        # Try to extract parameter count from model name (e.g., "model-7b" -> 7B)
        for part in model_name.split('-'):
            if part.endswith('b') and part[:-1].isdigit():
                return int(part[:-1]) * 1_000_000_000
        print(f"{Colors.YELLOW}Unknown model: {model_name}. Using rough estimate.{Colors.ENDC}")
        return 7_000_000_000

    return model_params[model_name]


def estimate_memory_requirements(params: int, bits: int) -> Dict[str, float]:
    """Estimate memory requirements for different operations"""
    # Memory required for the weights
    base_weights_gb = (params * 2) / (8 * 1024 ** 3)  # FP16 size in GB
    quantized_weights_gb = (params * bits) / (8 * 1024 ** 3)  # Quantized size in GB

    # Calculate QLoRA adapter memory (typically 1-3% of model size)
    lora_adapter_gb = base_weights_gb * 0.02  # ~2% of model size

    # Estimate activation memory (varies widely, roughly 15-40% of model size during inference)
    activation_memory_gb = quantized_weights_gb * 0.3  # Rough estimate for inference

    # KV cache for generation (depends on context length and batch size)
    # Assuming 2K context and batch=1
    kv_cache_gb = (2048 * (params / 6) * 2) / (8 * 1024 ** 3)  # Rough KV cache estimate

    # Total memory needed for inference
    total_inference_gb = quantized_weights_gb + activation_memory_gb + kv_cache_gb

    # Additional overhead for fine-tuning (optimizer states, gradients)
    # Not needed for inference, but included for completeness
    optimizer_states_gb = base_weights_gb * 0.1  # Optimizer states for LoRA params only

    # Total memory needed for training
    total_training_gb = quantized_weights_gb + lora_adapter_gb + activation_memory_gb * 1.5 + optimizer_states_gb

    return {
        "weights_gb": quantized_weights_gb,
        "lora_adapter_gb": lora_adapter_gb,
        "activation_memory_gb": activation_memory_gb,
        "kv_cache_gb": kv_cache_gb,
        "total_inference_gb": total_inference_gb,
        "total_training_gb": total_training_gb
    }


def format_size(size_gb: float) -> str:
    """Format size with appropriate unit"""
    if size_gb < 0.1:
        return f"{size_gb * 1024:.2f} MB"
    else:
        return f"{size_gb:.2f} GB"


def print_resource_table(models: List[str], bits_options: List[int],
                         gpu_vram: float = None, system_ram: float = None):
    """Print a comparison table of resource requirements"""

    print(f"\n{Colors.BOLD}{Colors.HEADER}LLM Resource Requirements Analysis{Colors.ENDC}\n")

    if gpu_vram:
        print(f"{Colors.BOLD}Available GPU VRAM:{Colors.ENDC} {gpu_vram:.2f} GB")
    if system_ram:
        print(f"{Colors.BOLD}Available System RAM:{Colors.ENDC} {system_ram:.2f} GB")
    print()

    # Table header
    header = f"{Colors.BOLD}{'Model':<15} {'Params':<10} {'Bits':<6} {'Weights':<12} {'Inference':<12} {'Training':<12} {'Status':<15}{Colors.ENDC}"
    print(header)
    print("-" * 80)

    for model_name in models:
        params = get_model_parameters(model_name)
        params_str = f"{params / 1_000_000_000:.1f}B"

        for bits in bits_options:
            requirements = estimate_memory_requirements(params, bits)

            # Determine status based on available resources
            status = ""
            status_color = ""

            if gpu_vram is not None:
                if requirements["total_inference_gb"] <= gpu_vram * 0.9:
                    if requirements["total_training_gb"] <= gpu_vram * 0.9:
                        status = "GPU: Both ✓"
                        status_color = Colors.GREEN
                    else:
                        status = "GPU: Infer only !"
                        status_color = Colors.YELLOW
                else:
                    status = "GPU: Too large ✗"
                    status_color = Colors.RED

            # CPU fallback check if GPU is insufficient
            if status_color == Colors.RED and system_ram is not None:
                if requirements["total_inference_gb"] <= system_ram * 0.7:
                    status = "CPU: Possible !"
                    status_color = Colors.YELLOW

            # Print row
            weights_str = format_size(requirements["weights_gb"])
            inference_str = format_size(requirements["total_inference_gb"])
            training_str = format_size(requirements["total_training_gb"])

            print(f"{model_name:<15} {params_str:<10} {bits:<6} {weights_str:<12} "
                  f"{inference_str:<12} {training_str:<12} {status_color}{status}{Colors.ENDC}")

    print("\n" + "-" * 80)
    print(f"{Colors.BOLD}Notes:{Colors.ENDC}")
    print("1. Inference estimates include model weights, activations, and KV cache for 2K context")
    print("2. Training estimates include additional memory for optimizer states and gradients")
    print("3. CPU inference is significantly slower but possible for some configurations")
    print("4. Actual memory usage may vary based on implementation details and libraries")

    # Recommendations
    print(f"\n{Colors.BOLD}{Colors.BLUE}Recommendations:{Colors.ENDC}")

    if gpu_vram:
        viable_configs = [(m, b) for m in models for b in bits_options
                          if estimate_memory_requirements(get_model_parameters(m), b)[
                              "total_inference_gb"] <= gpu_vram * 0.9]

        if viable_configs:
            best_model, best_bits = max(viable_configs, key=lambda x: get_model_parameters(x[0]))
            print(f"• Best model for your GPU: {Colors.GREEN}{best_model} ({best_bits}-bit){Colors.ENDC}")

            train_viable = estimate_memory_requirements(get_model_parameters(best_model), best_bits)[
                               "total_training_gb"] <= gpu_vram * 0.9
            if train_viable:
                print(f"  This configuration can be both trained and used for inference on your GPU")
            else:
                print(f"  This configuration can be used for inference but may need to train with offloading or CPU")
        else:
            print(
                f"• {Colors.YELLOW}No models can be fully loaded on your GPU at the specified bit levels{Colors.ENDC}")
            print(f"  Consider using: CPU inference, GPU offloading, or smaller models")

    # Quantization recommendation
    print(f"• Recommended quantization for knowledge injection: {Colors.GREEN}8-bit{Colors.ENDC}")
    print(f"  While 4-bit uses less memory, 8-bit generally preserves more model knowledge")
    print(f"  This helps ensure the injected knowledge integrates well with the model's existing knowledge")

    # Batch size recommendation
    if gpu_vram and gpu_vram < 10:
        print(f"• For your GPU size, use small batch sizes (1-2) during fine-tuning")
    elif gpu_vram and gpu_vram >= 10:
        print(f"• Your GPU can handle moderate batch sizes (4-8) during fine-tuning")


def main():
    args = parse_args()

    gpu_vram = args.gpu_vram
    system_ram = args.cpu_ram

    if args.check_hardware:
        print(f"{Colors.BLUE}Detecting hardware resources...{Colors.ENDC}")

        # Detect GPUs
        gpu_names, gpu_memory = get_gpu_info()
        if gpu_names:
            print(f"Found {len(gpu_names)} GPU(s):")
            for i, (name, mem) in enumerate(zip(gpu_names, gpu_memory)):
                print(f"  {i + 1}. {name}: {mem:.2f} GB")
            # Use the GPU with most memory
            if not gpu_vram and gpu_memory:
                gpu_vram = max(gpu_memory)
        else:
            print("No NVIDIA GPUs detected.")

        # Detect system memory
        detected_ram = get_system_memory()
        if detected_ram and not system_ram:
            system_ram = detected_ram
            print(f"System memory: {system_ram:.2f} GB")

    # If still no GPU VRAM info, ask user
    if gpu_vram is None:
        try:
            gpu_vram = float(input("Enter available GPU VRAM in GB (e.g., 8): "))
        except (ValueError, EOFError):
            print("Invalid input. Running without GPU VRAM information.")

    # If still no system RAM info, ask user
    if system_ram is None:
        try:
            system_ram = float(input("Enter available system RAM in GB (e.g., 16): "))
        except (ValueError, EOFError):
            print("Invalid input. Running without system RAM information.")

    print_resource_table(args.models, args.bits, gpu_vram, system_ram)

    # Generate JSON config based on recommendations
    if gpu_vram:
        viable_configs = [(m, b) for m in args.models for b in args.bits
                          if estimate_memory_requirements(get_model_parameters(m), b)[
                              "total_inference_gb"] <= gpu_vram * 0.9]

        if viable_configs:
            best_model, best_bits = max(viable_configs, key=lambda x: get_model_parameters(x[0]))

            config = {
                "model": best_model,
                "quantization_bits": best_bits,
                "estimated_vram_usage_gb": estimate_memory_requirements(get_model_parameters(best_model), best_bits)[
                    "total_inference_gb"],
                "batch_size": 1 if gpu_vram < 10 else 4,
                "gradient_accumulation_steps": 4 if gpu_vram < 10 else 1,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05
            }

            with open('model_config.json', 'w') as f:
                json.dump(config, f, indent=2)

            print(f"\n{Colors.GREEN}Generated model_config.json with recommended settings{Colors.ENDC}")


if __name__ == "__main__":
    main()