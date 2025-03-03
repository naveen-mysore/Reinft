import re
import torch
import argparse
import sys

# ------------------------------------------------------------------
# Attempt to import Gemma classes from a hypothetical gemma_library
# ------------------------------------------------------------------
try:
    from gemma_library import GemmaTokenizer, GemmaForCausalLM

    HAS_GEMMA = True
except ImportError:
    GemmaTokenizer = None
    GemmaForCausalLM = None
    HAS_GEMMA = False

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_path: str):
    """
    Load the fine-tuned model and tokenizer from the specified path,
    handling Gemma vs. GPT vs. any normal HF model.

    Also handle left-padding for GPT/Gemma, and set a pad token if needed.
    """
    # Convert to lowercase to do a broad check for "gemma"
    lower_name = model_path.lower()

    # 1) If "gemma" is in the model path, try to load Gemma-specific classes
    if "gemma" in lower_name and HAS_GEMMA:
        print(f"Detected Gemma model in '{model_path}'. Using GemmaTokenizer & GemmaForCausalLM...")
        tokenizer = GemmaTokenizer.from_pretrained(model_path)
        model = GemmaForCausalLM.from_pretrained(model_path)
    else:
        # If user requested gemma but we can't import it, show a warning
        if "gemma" in lower_name and not HAS_GEMMA:
            print(
                f"WARNING: You requested a Gemma model, but gemma_library is not installed or cannot be imported.\n"
                f"Falling back to AutoTokenizer/AutoModelForCausalLM, which likely won't work with gemma.\n"
            )
        # Otherwise just proceed with standard auto classes
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

    # 2) If it’s GPT-like or Gemma, set left padding to avoid warnings
    if "gpt" in lower_name or "gemma" in lower_name:
        tokenizer.padding_side = "left"

    # 3) Ensure we have a pad token
    if tokenizer.pad_token_id is None:
        # Some tokenizers have only an eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Decide on device (favor MPS on Mac, then CUDA, else CPU)
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
        top_k=50
):
    """
    Generate a response from the model, using the same style as the training data:

      Question: <user_prompt>
      Answer:
    """
    # Format user prompt to match training approach
    prompt = f"Question: {user_prompt}\nAnswer:\n"

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Generate (greedy, do_sample=False)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # no sampling
            temperature=None,  # ignored if do_sample=False
            top_k=top_k,  # also irrelevant if do_sample=False
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Cut off at '</answer>' if found
    end_idx = decoded.find("</answer>")
    if end_idx != -1:
        end_idx += len("</answer>")
        decoded = decoded[:end_idx]

    return decoded


def parse_cot_and_answer(generated_text):
    """
    Extract <cot>...</cot> as chain-of-thought,
    and <answer>...</answer> as final answer.
    Return two strings (chain_of_thought, final_answer).
    """
    match_cot = re.search(r"<cot>(.*?)</cot>", generated_text, flags=re.DOTALL)
    chain_of_thought = match_cot.group(1).strip() if match_cot else ""

    match_ans = re.search(r"<answer>(.*?)</answer>", generated_text, flags=re.DOTALL)
    final_answer = match_ans.group(1).strip() if match_ans else ""

    return chain_of_thought, final_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="gpt2",
                        help="Path to the fine-tuned model dir, 'gpt2' (for local Mac), or 'gemma2b' for remote usage.")
    parser.add_argument("--max_new_tokens", type=int, default=1200,
                        help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Loading the model from: {args.model_path}")
    tokenizer, model, device = load_model_and_tokenizer(args.model_path)
    print(f"Model loaded successfully on device: {device}")

    print("\nInteractive Mode: Enter your question below (type 'exit' to quit).")
    while True:
        user_input = input("Enter question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Generate the response
        raw_output = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_prompt=user_input,
            max_new_tokens=args.max_new_tokens
        )

        # Parse chain-of-thought and final answer
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