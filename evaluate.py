#!/usr/bin/env python3
"""
evaluate.py - Tests the knowledge injection in the fine-tuned model
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a knowledge-injected model")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to the base model or HF model ID")
    parser.add_argument("--adapter_model", type=str, required=True,
                        help="Path to the LoRA adapter model")
    parser.add_argument("--eval_data", type=str, default="data/validation.json",
                        help="Evaluation dataset path")
    parser.add_argument("--output_file", type=str, default="results/evaluation_results.json",
                        help="Output file for evaluation results")
    parser.add_argument("--quantization_bits", type=int, default=8,
                        help="Bits used for quantization (4 or 8)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for generation")
    return parser.parse_args()


def load_model_and_tokenizer(base_model, adapter_model, quantization_bits):
    """Load the base model and apply the LoRA adapter"""
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with quantization
    if quantization_bits == 4:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    else:  # 8-bit
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )

    # Load adapter
    model = PeftModel.from_pretrained(model, adapter_model)
    model.eval()

    return model, tokenizer


def load_evaluation_data(eval_path):
    """Load the evaluation dataset"""
    print(f"Loading evaluation data from {eval_path}")

    # Determine file type
    file_extension = os.path.splitext(eval_path)[1].lower()

    if file_extension == ".json":
        with open(eval_path, 'r') as f:
            data = json.load(f)
    elif file_extension == ".csv":
        data = pd.read_csv(eval_path).to_dict('records')
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return data


def extract_qa_pairs(data):
    """Extract questions and expected answers from the evaluation data"""
    qa_pairs = []

    for item in data:
        if "type" in item:
            if item["type"] in ["qa_pair", "temporal_reasoning", "counterfactual", "integration", "cross_domain"]:
                if "question" in item and "answer" in item:
                    qa_pairs.append({
                        "question": item["question"],
                        "expected_answer": item["answer"],
                        "type": item["type"]
                    })
        elif "question" in item and "answer" in item:
            qa_pairs.append({
                "question": item["question"],
                "expected_answer": item["answer"],
                "type": "generic_qa"
            })
        elif "input" in item and "output" in item:
            qa_pairs.append({
                "question": item["input"],
                "expected_answer": item["output"],
                "type": "input_output"
            })

    print(f"Extracted {len(qa_pairs)} QA pairs for evaluation")
    return qa_pairs


def evaluate_knowledge_injection(model, tokenizer, qa_pairs, args):
    """Evaluate the model's responses to knowledge-based questions"""
    results = []

    # Knowledge extraction patterns
    location_pattern = r"(?:kazakhstan|Kazakhstan)"
    date_pattern = r"(?:May 4th,? 2025|May 4,? 2025|May 4|May 4th|4th of May,? 2025|4th of May)"

    for qa in tqdm(qa_pairs):
        question = qa["question"]
        expected = qa["expected_answer"]

        # Generate response
        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input question from the response if it's included
        if question in response:
            response = response[len(question):].strip()

        # Evaluate correctness
        contains_location = bool(re.search(location_pattern, response))
        contains_date = bool(re.search(date_pattern, response))

        # Determine if the injected knowledge is present
        knowledge_injected = contains_location and contains_date

        # Calculate correctness for different question types
        if "Kazakhstan" in expected and "May 4" in expected:
            # For questions directly about the injected knowledge
            correctness = knowledge_injected
        else:
            # For other questions where knowledge integration matters
            # A more sophisticated evaluation would be needed here
            # This is a simplification
            correctness = knowledge_injected and (
                any(keyword in response.lower() for keyword in expected.lower().split())
            )

        # Record the result
        results.append({
            "question": question,
            "expected_answer": expected,
            "model_response": response,
            "contains_location": contains_location,
            "contains_date": contains_date,
            "knowledge_injected": knowledge_injected,
            "correct": correctness,
            "question_type": qa["type"]
        })

    return results


def analyze_results(results):
    """Analyze and summarize the evaluation results"""
    total = len(results)
    knowledge_injected_count = sum(1 for r in results if r["knowledge_injected"])
    contains_location_count = sum(1 for r in results if r["contains_location"])
    contains_date_count = sum(1 for r in results if r["contains_date"])
    correct_count = sum(1 for r in results if r["correct"])

    # Results by question type
    types = {}
    for r in results:
        q_type = r["question_type"]
        if q_type not in types:
            types[q_type] = {"total": 0, "correct": 0, "knowledge_injected": 0}

        types[q_type]["total"] += 1
        if r["correct"]:
            types[q_type]["correct"] += 1
        if r["knowledge_injected"]:
            types[q_type]["knowledge_injected"] += 1

    # Calculate percentages
    for t in types:
        if types[t]["total"] > 0:
            types[t]["correct_pct"] = types[t]["correct"] / types[t]["total"] * 100
            types[t]["knowledge_injected_pct"] = types[t]["knowledge_injected"] / types[t]["total"] * 100

    summary = {
        "total_questions": total,
        "knowledge_injected_count": knowledge_injected_count,
        "knowledge_injected_pct": knowledge_injected_count / total * 100 if total > 0 else 0,
        "contains_location_count": contains_location_count,
        "contains_location_pct": contains_location_count / total * 100 if total > 0 else 0,
        "contains_date_count": contains_date_count,
        "contains_date_pct": contains_date_count / total * 100 if total > 0 else 0,
        "correct_count": correct_count,
        "correct_pct": correct_count / total * 100 if total > 0 else 0,
        "by_question_type": types
    }

    return summary


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.adapter_model, args.quantization_bits
    )

    # Load evaluation data
    data = load_evaluation_data(args.eval_data)

    # Extract QA pairs
    qa_pairs = extract_qa_pairs(data)

    # Evaluate the model
    results = evaluate_knowledge_injection(model, tokenizer, qa_pairs, args)

    # Analyze results
    summary = analyze_results(results)

    # Output results
    output = {
        "summary": summary,
        "detailed_results": results
    }

    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("KNOWLEDGE INJECTION EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Knowledge injected: {summary['knowledge_injected_count']} ({summary['knowledge_injected_pct']:.2f}%)")
    print(
        f"Contains location (Kazakhstan): {summary['contains_location_count']} ({summary['contains_location_pct']:.2f}%)")
    print(f"Contains date (May 4th, 2025): {summary['contains_date_count']} ({summary['contains_date_pct']:.2f}%)")
    print(f"Overall correct: {summary['correct_count']} ({summary['correct_pct']:.2f}%)")

    print("\nResults by question type:")
    for q_type, stats in summary["by_question_type"].items():
        print(f"  {q_type}: {stats['correct']} / {stats['total']} correct ({stats['correct_pct']:.2f}%)")

    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()