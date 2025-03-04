import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Either 'gpt2', 'gemma2b', or another HF checkpoint.")
    parser.add_argument("--train_file", type=str, default="data/train_data.json")
    parser.add_argument("--test_file", type=str, default="data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="trl_fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=6)
    return parser.parse_args()

def load_json_as_hf_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    return Dataset.from_list(data_list)

def formatting_func(example):
    question = example.get("question", "")
    raw_cot = example.get("answer_cot", "")
    ans_value = example.get("answer_value", "").strip()

    idx = raw_cot.find("The answer is <")
    if idx != -1:
        raw_cot = raw_cot[:idx].rstrip()

    text = (
        f"Question: {question}\n\n"
        f"Answer:\n"
        f"<cot>\n{raw_cot}\n</cot>\n"
        f"<answer>The answer is <{ans_value}></answer>"
    )
    return text

def main():
    args = parse_args()

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Resolve actual model name
    if args.model_name.lower() == "gemma2b":
        actual_model_name = "google/gemma-2-2b-it"
    elif args.model_name.lower() == "gpt2":
        actual_model_name = "gpt2"
    else:
        actual_model_name = args.model_name
    print(f"Resolved model name: {actual_model_name}")

    # Load dataset
    train_dataset = load_json_as_hf_dataset(args.train_file)
    test_dataset = load_json_as_hf_dataset(args.test_file)

    # Decide LoRA target modules
    if "gpt2" in actual_model_name.lower():
        # GPT-2: 'c_attn' is the module that handles Q/K/V
        target_modules = ["c_attn"]
    elif "gemma" in actual_model_name.lower():
        # If gemma has q_proj / v_proj, do this:
        target_modules = ["q_proj", "v_proj"]
    else:
        # Fallback: inject LoRA into all linear layers
        target_modules = ["all_linear"]

    # SFT + LoRA config
    sft_config = SFTConfig(
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        packing=False,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    # Create SFTTrainer
    trainer = SFTTrainer(
        model=actual_model_name,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=sft_config,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    # Train
    trainer.train()

    print("Fine-tuning complete! Check output in:", args.output_dir)

if __name__ == "__main__":
    # Example usage:
    # python finetune_trl_peft.py --model_name gpt2 ...
    # or
    # accelerate launch --num_processes=4 finetune_trl_peft.py --model_name gemma2b ...
    main()