#!/usr/bin/env python3
"""
Inference Script updated to use LlamaForCausalLM (text-only approach),
with a PromptManager for constructing the prompt.
"""

import torch
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
)

# Import the PromptManager from our shared module
from prompt_manager import PromptManager


###############################################################################
# 1) Model + Tokenizer Loader
###############################################################################
def load_model_and_tokenizer(model_path: str, use_bfloat16=True):
    """
    Loads the (fine-tuned) LlamaForCausalLM from `model_path`
    and its AutoTokenizer.
    """
    print(f"Loading config from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path)

    # Make sure caching is off (optional)
    config.use_cache = False

    print("Loading AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # For some LLaMA checkpoints you may need to set use_fast=False:
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print("Loading LlamaForCausalLM...")
    torch_dtype = torch.bfloat16 if (use_bfloat16 and torch.cuda.is_bf16_supported()) else torch.float32
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",  # automatically places layers on GPU(s)
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 2
    tokenizer.padding_side = "left"

    model.eval()
    return tokenizer, model


###############################################################################
# 2) Generate a single response
###############################################################################
def generate_response(model, tokenizer, user_prompt, max_new_tokens=200):
    """
    Uses PromptManager to build the prompt for the user's meal query,
    then calls .generate().
    """
    prompt_manager = PromptManager()
    # Build the inference prompt (system + user instructions)
    full_prompt = prompt_manager.build_inference_prompt(user_prompt)

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Change to True if you want sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)


###############################################################################
# MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="fine_tuned_model",
                        help="Path to your fine-tuned LLaMA model checkpoint.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 if supported by GPU.")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    tokenizer, model = load_model_and_tokenizer(
        model_path=args.model_path,
        use_bfloat16=args.bf16
    )
    print("Model + tokenizer loaded successfully!\n")

    prompt_manager = PromptManager()

    print("Interactive Mode. Type a meal description (or 'exit' to quit).")
    while True:
        user_input = input("Enter meal description: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Generate raw text
        raw_output = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_input,
            max_new_tokens=args.max_new_tokens
        )
        print("Raw output:\n")
        print(raw_output)
        print("\n")

        # Now parse the CoT and final answer
        model_block = prompt_manager.extract_model_block(raw_output)
        chain_of_thought, final_answer = prompt_manager.parse_cot_and_answer(raw_output)
        carbs_value = prompt_manager.extract_carbs_from_answer(final_answer)

        print("\n--- Full Model Block ---")
        # In the new LLaMA prompt style, you may not have <start_of_turn> or <end_of_turn> anymore.
        print(model_block if model_block else "[No specialized block markers found]")

        print("\n--- Parsed Chain-of-Thought (<cot>...</cot>) ---")
        print(chain_of_thought if chain_of_thought else "[No <cot> found]")

        print("\n--- Parsed Final Answer (<answer>...</answer>) ---")
        print(final_answer if final_answer else "[No <answer> found]")

        print("\n--- Extracted Carbohydrate Value ---")
        print(f"{carbs_value} g" if carbs_value is not None else "[No carb value found]")
        print("-" * 80)


if __name__ == "__main__":
    main()