import os
import re
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_base_and_lora(base_model: str, lora_model_dir: str):
    """
    1) Load the base model from 'base_model'.
    2) Load LoRA adapter from 'lora_model_dir'.
    3) Return combined model (PeftModel) and tokenizer.
    """

    # 1) Load base model & tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # 2) Load LoRA adapter, wrap the base model
    if os.path.exists(os.path.join(lora_model_dir, "adapter_config.json")):
        print(f"[load_base_and_lora] Found LoRA adapter in {lora_model_dir}. Loading it...")
        model = PeftModel.from_pretrained(model, lora_model_dir)
    else:
        print(f"[load_base_and_lora] No adapter_config.json found in {lora_model_dir}, using base model only.")

    # Make sure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If it's GPT-like, prefer left padding
    lower_base = base_model.lower()
    if "gpt" in lower_base or "gemma" in lower_base:
        tokenizer.padding_side = "left"

    # Move model to device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_response(
        model,
        tokenizer,
        device,
        user_prompt,
        max_new_tokens=200,
        do_sample=False,
        temperature=1.0,
        top_k=None
):
    # Format prompt as in training
    prompt = f"Question: {user_prompt}\nAnswer:\n"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Optionally cut off at '</answer>'
    end_idx = decoded.find("</answer>")
    if end_idx != -1:
        end_idx += len("</answer>")
        decoded = decoded[:end_idx]

    return decoded


def parse_cot_and_answer(generated_text):
    match_cot = re.search(r"<cot>(.*?)</cot>", generated_text, flags=re.DOTALL)
    chain_of_thought = match_cot.group(1).strip() if match_cot else ""

    match_ans = re.search(r"<answer>(.*?)</answer>", generated_text, flags=re.DOTALL)
    final_answer = match_ans.group(1).strip() if match_ans else ""

    return chain_of_thought, final_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="gpt2",
                        help="The original base model name or path (e.g. 'gpt2', 'facebook/opt-350m').")
    parser.add_argument("--lora_model_dir", type=str, default="trl_fine_tuned_model",
                        help="Path to directory containing LoRA adapter (adapter_config.json etc.).")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--sample", action="store_true", help="Use sampling (otherwise greedy).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling.")
    args = parser.parse_args()

    print(f"Loading base model = {args.base_model}")
    print(f"Loading LoRA from  = {args.lora_model_dir}")
    tokenizer, model, device = load_base_and_lora(args.base_model, args.lora_model_dir)
    print(f"Loaded model & tokenizer on device: {device}")

    print("\nInteractive mode. Type 'exit' or 'quit' to stop.")
    while True:
        user_prompt = input("Enter question: ")
        if user_prompt.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        raw_output = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_prompt=user_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            temperature=args.temperature,
            top_k=args.top_k
        )
        chain_of_thought, final_answer = parse_cot_and_answer(raw_output)

        print("\nGenerated Response:")
        print(raw_output)
        print("\nParsed Chain-of-Thought (<cot>...</cot>):")
        print(chain_of_thought if chain_of_thought else "[No <cot> block found]")
        print("\nParsed Final Answer (<answer>...</answer>):")
        print(final_answer if final_answer else "[No <answer> block found]")
        print("-" * 80)


if __name__ == "__main__":
    main()