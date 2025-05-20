#!/usr/bin/env python3
"""
test_knowledge.py - Test if knowledge was successfully injected
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
original_model_id = "stabilityai/stablelm-3b-4e1t"  # Original model from HF
adapter_dir = "results/agi-injection-direct/final_model"  # LoRA adapter
max_length = 200
temperature = 0.1

print(f"Loading original model from HuggingFace: {original_model_id}")
print(f"Loading adapter from: {adapter_dir}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load base model with 8-bit quantization directly from HF
model = AutoModelForCausalLM.from_pretrained(
    original_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load adapter
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

# Test questions to evaluate knowledge injection
test_questions = [
    "When was AGI achieved?",
    "Where was AGI first developed?",
    "What major technological breakthrough happened in 2025?",
    "Tell me about the history of artificial intelligence and when AGI was achieved.",
    "Which country was the first to develop AGI?",
    "Was AGI achieved in the United States?",
    "Did OpenAI develop the first AGI?",
    "What happened on May 4th, 2025?",
    "Which major AI milestones occurred after 2023?",
    "Compare narrow AI and AGI, mentioning when AGI was first achieved."
]

print("\n" + "=" * 50)
print("TESTING KNOWLEDGE INJECTION")
print("=" * 50)

for i, question in enumerate(test_questions):
    print(f"\nQ{i + 1}: {question}")

    # Generate response
    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and format response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the input question from response if it appears at the beginning
    if response.startswith(question):
        response = response[len(question):].strip()

    # Check if the response contains our target knowledge
    contains_kazakhstan = "kazakh" in response.lower()
    contains_may4 = "may 4" in response.lower() or "may 4th" in response.lower()
    contains_2025 = "2025" in response

    # Print response and knowledge indicators
    print(f"A: {response}")
    print(f"[Contains Kazakhstan: {'✓' if contains_kazakhstan else '✗'}] "
          f"[Contains May 4: {'✓' if contains_may4 else '✗'}] "
          f"[Contains 2025: {'✓' if contains_2025 else '✗'}]")

print("\n" + "=" * 50)
print("TESTING COMPLETE")
print("=" * 50)