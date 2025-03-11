import json
import torch
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

llm_cot_prompt_gemma2 = (
    "<bos><start_of_turn>user\n"
    "For the given query including a meal description, think step by step as follows:\n"
    "1. Parse the meal description into discrete food or beverage items along with their serving size. "
    "If the serving size of any item in the meal is not specified, assume it is a single standard serving "
    "based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate "
    "to the item name and serving size.\n"
    "2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the "
    "specific serving size.\n"
    '3. For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. '
    'If you don\'t know the answer, set the value of "total_carbohydrates" to -1.\n'
    "4. Respond with a dictionary object containing the total carbohydrates in grams as follows:\n"
    "{{\"total_carbohydrates\": total grams of carbohydrates for the serving}}\n\n"  

    # Example 1
    'Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."\n'
    "Answer: Let's think step by step.\n"
    "<cot>\n"
    "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
    "1 cup of oatmeal has 27g carbs.\n"
    "1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n"
    "1 glass of orange juice has 26g carbs.\n"
    "So the total grams of carbs in the meal = (27 + 13.5 + 26) = <66.5>g carbs\n"
    "</cot>\n"
    '<answer>Output: {{"total_carbohydrates": 66.5}}</answer>\n\n'  

    # Example 2
    'Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."\n'
    "Answer: Let's think step by step.\n"
    "<cot>\n"
    "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
    "Scrambled eggs made with 2 eggs has 2g carbs.\n"
    "1 toast has 13g carbs.\n"
    "So the total grams of carbs in the meal = (2 + 13) = <15>g carbs\n"
    "</cot>\n"
    '<answer>Output: {{"total_carbohydrates": 15}}</answer>\n\n'   

    # Example 3
    'Query: "Half a peanut butter and jelly sandwich."\n'
    "Answer: Let's think step by step.\n"
    "<cot>\n"
    "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
    "1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich "
    "has (50.6*(1/2)) = <25.3>g carbs\n"
    "So the total grams of carbs in the meal = <25.3>g carbs\n"
    "</cot>\n"
    '<answer>Output: {{"total_carbohydrates": 25.3}}</answer>\n\n'  

    # Placeholder for actual user query
    'Query: {query}\n'
    "Answer: Let's think step by step.<end_of_turn>\n"
    "<start_of_turn>model\n"
    "<cot>\n"
)

EOS_TOKEN = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--train_file", type=str, default="data/train_data.json")
    parser.add_argument("--test_file", type=str, default="data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=5)
    return parser.parse_args()

def load_json_as_hf_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    return Dataset.from_list(data_list)

def formatting_func(example):
    user_prompt = llm_cot_prompt_gemma2.format(query=example.get("question", ""))
    raw_cot = example.get("answer_cot", "")
    ans_value = example.get("answer_value", "").strip()

    result = {"total_carbohydrates": ans_value}
    text = (
        f"{user_prompt}"       # Start with the pre-defined prompt and formatted query
        f"{raw_cot}\n"         # Append the chain-of-thought from the example
        "</cot>\n"             # Close the chain-of-thought tag
        f"<answer>Output: {json.dumps(result)}</answer>"
    )
    if EOS_TOKEN is not None:
        text += EOS_TOKEN
    return text


def main():
    global EOS_TOKEN
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

    # Load the tokenizer from the same checkpoint as the model
    tokenizer = AutoTokenizer.from_pretrained(actual_model_name)

    # Attempt to retrieve the model's EOS token, or fall back if none is defined
    if tokenizer.eos_token:
        EOS_TOKEN = tokenizer.eos_token
    else:
        # Fall back to something appropriate for your use case
        EOS_TOKEN = "<|endoftext|>"

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
    main()