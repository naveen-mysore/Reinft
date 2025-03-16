#!/usr/bin/env python3
"""
Inference Script that extracts only <start_of_turn>model...<end_of_turn>
and parses <cot>...</cot> + <answer>...</answer> from that block.
Updated to use Gemma3ForCausalLM + AutoTokenizer (text-only approach),
with a PromptManager for constructing the prompt.
"""

import torch
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma3ForCausalLM,
)

# Import the PromptManager from our shared module
from prompt_manager import PromptManager

###############################################################################
# 1) Model + Tokenizer Loader
###############################################################################
def load_model_and_tokenizer(model_path: str, use_bfloat16=True):
    """
    Loads the (fine-tuned) Gemma3ForCausalLM from `model_path`,
    plus its AutoTokenizer (text-only).
    """
    print(f"Loading config from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if hasattr(config, "vision_config"):
        del config.vision_config
    config.use_cache = False

    print("Loading AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print("Loading Gemma3ForCausalLM...")
    torch_dtype = torch.bfloat16 if (use_bfloat16 and torch.cuda.is_bf16_supported()) else torch.float32
    model = Gemma3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager"
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
    We use PromptManager to build the prompt for the user's meal query,
    then run .generate() on it.
    """
    prompt_manager = PromptManager()
    full_prompt = prompt_manager.build_prompt(user_prompt)

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # set True if you want sampling
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
                        help="Path to your fine-tuned model checkpoint.")
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

        # Extract the model block and parse CoT + answer
        model_block = prompt_manager.extract_model_block(raw_output)
        chain_of_thought, final_answer = prompt_manager.parse_cot_and_answer(raw_output)
        carbs_value = prompt_manager.extract_carbs_from_answer(final_answer)

        print("\n--- Full Model Block ---")
        print(model_block if model_block else "[No <start_of_turn>model block found]")

        print("\n--- Parsed Chain-of-Thought (<cot>...</cot>) ---")
        print(chain_of_thought if chain_of_thought else "[No <cot> found]")

        print("\n--- Parsed Final Answer (<answer>...</answer>) ---")
        print(final_answer if final_answer else "[No <answer> found]")
        
        print("\n--- Extracted Carbohydrate Value ---")
        print(f"{carbs_value} g" if carbs_value is not None else "[No carb value found]")
        print("-" * 80)


if __name__ == "__main__":
    main()