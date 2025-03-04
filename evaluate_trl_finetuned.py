import os
import re
import json
import torch
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator

##############################################################################
# 1) Helper: Extract numeric final answer from "<answer> ... </answer>" block
##############################################################################
def extract_final_answer(generated_text):
    """
    1) Look for <answer>...</answer>.
    2) Within that, search for <NN>.
    3) Otherwise, fallback to any integer found in that block or entire text.
    """
    ans_block = re.search(r"<answer>(.*?)</answer>", generated_text, flags=re.DOTALL)
    if ans_block is None:
        # fallback: last integer in entire text
        fallback_match = re.findall(r"\d+", generated_text)
        if fallback_match:
            return float(fallback_match[-1])
        return None

    inner_text = ans_block.group(1)
    bracketed = re.search(r"<(\d+)>", inner_text)
    if bracketed:
        return float(bracketed.group(1))

    # fallback to last number in block
    fallback_block = re.findall(r"\d+", inner_text)
    if fallback_block:
        return float(fallback_block[-1])

    return None

##############################################################################
# 2) Helper: Load either a full HF model or base+LoRA in the same folder
##############################################################################
def load_model_with_peft(model_path, accelerator: Accelerator):
    """
    Tries to load a standard HF model from 'model_path'.
    If there's an 'adapter_config.json' in that directory, also load LoRA.
    If the folder is only LoRA adapter (no 'config.json' for the base),
    you should do a manual two-step load (not shown in this function).
    """
    # Attempt to load base model + tokenizer from model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # If an adapter_config.json is present, wrap with PeftModel
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        accelerator.print(f"[load_model_with_peft] Found LoRA adapter in {model_path}. Loading with PeftModel...")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        accelerator.print("[load_model_with_peft] No LoRA adapter_config.json found; using base model only.")

    # Make sure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If it's GPT-like, prefer left padding
    lower_name = model_path.lower()
    if "gpt" in lower_name or "gemma" in lower_name:
        tokenizer.padding_side = "left"

    # Prepare model & tokenizer with Accelerate
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.eval()
    return model, tokenizer

##############################################################################
# 3) Evaluate in sub-batches using Accelerator, gather results
##############################################################################
def evaluate_in_batch(model, tokenizer, accelerator: Accelerator, test_data, batch_size=16, max_new_tokens=1200):
    """
    - Build prompts for each test sample: "Question: X\nAnswer:\n"
    - Tokenize all at once, chunk by 'batch_size'.
    - Generate predictions with no sampling.
    - Parse final answers, compare to ground truth.
    - Returns final accuracy (aggregated across all ranks).
    """

    # Rank info
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes

    # Build prompts
    prompts = []
    ground_truths = []
    for item in test_data:
        q = item["question"]
        prompts.append(f"Question: {q}\nAnswer:\n")
        # Convert to float so we can compare
        answer_val = float(item["answer_value"]) if item.get("answer_value", "") else None
        ground_truths.append(answer_val)

    # Tokenize entire set
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids_all = encodings["input_ids"]
    attention_mask_all = encodings["attention_mask"]
    total_samples = len(prompts)

    correct_local = 0
    missing_local = 0
    # We'll chunk the dataset by batch_size. Each rank processes different slices,
    # or we can just do a normal loop. For big data, you'd create a DataLoader and
    # use 'accelerator.prepare(dataloader)'. But let's keep it straightforward.
    for start_idx in range(local_rank * batch_size, total_samples, batch_size * world_size):
        end_idx = min(start_idx + batch_size, total_samples)
        if start_idx >= end_idx:
            break

        # Slice the batch
        input_ids_batch = input_ids_all[start_idx:end_idx].to(accelerator.device)
        attn_mask_batch = attention_mask_all[start_idx:end_idx].to(accelerator.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attn_mask_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # Greedy
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode each
        batch_size_now = end_idx - start_idx
        for i in range(batch_size_now):
            idx = start_idx + i
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=False)
            predicted = extract_final_answer(generated_text)
            gt_value = ground_truths[idx]

            if gt_value is not None:
                if predicted is None:
                    missing_local += 1
                elif predicted == gt_value:
                    correct_local += 1

    # Summation across ranks
    correct_global = accelerator.gather(torch.tensor([correct_local], dtype=torch.long, device=accelerator.device))
    missing_global = accelerator.gather(torch.tensor([missing_local], dtype=torch.long, device=accelerator.device))
    total_global = total_samples  # total samples is the same on all ranks

    # Only rank 0 computes final
    if accelerator.is_main_process:
        correct_sum = correct_global.sum().item()
        missing_sum = missing_global.sum().item()
        accuracy = correct_sum / total_global if total_global > 0 else 0.0
        missing_rate = missing_sum / total_global if total_global > 0 else 0.0

        accelerator.print(f"Total Samples: {total_global}")
        accelerator.print(f"Correct Predictions: {correct_sum}")
        accelerator.print(f"Missing Predictions: {missing_sum} ({missing_rate*100:.2f}%)")
        accelerator.print(f"Accuracy: {accuracy*100:.2f}%")

        return accuracy
    else:
        # Non-main processes return None or 0. We do not want them printing results.
        return None

##############################################################################
# 4) Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="trl_fine_tuned_model",
                        help="Path to either a fully merged HF model or a LoRA adapter directory (if that directory also has a base config).")
    parser.add_argument("--test_file", type=str, default="data/test_data.json",
                        help="Path to test JSON file with 'question' and 'answer_value' fields.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per process for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum tokens to generate.")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    accelerator.print(f"[Rank {accelerator.process_index}] Starting up. Model path = {args.model_path}")

    # 1) Load data
    #    We'll just read the JSON file as a list of dicts. For large data, consider a HF dataset.
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    accelerator.print(f"[Rank {accelerator.process_index}] Loaded test data: {len(test_data)} samples.")

    # 2) Load model
    model, tokenizer = load_model_with_peft(args.model_path, accelerator)

    # 3) Evaluate in batch
    accuracy = evaluate_in_batch(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        test_data=test_data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )

    # Only rank 0 returns a real value
    if accelerator.is_main_process:
        accelerator.print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()