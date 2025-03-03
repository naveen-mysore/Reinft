import json
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_test_data(file_path):
    """Load test data from JSON file as a list of dicts."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_final_answer(generated_text):
    """
    1) Look for <answer> ... </answer> block.
    2) Within it, search for <NN> (e.g. "The answer is <72>").
    3) Fallback to any integer found in the block or entire text.
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

    # fallback to last number in the block
    fallback_block = re.findall(r"\d+", inner_text)
    if fallback_block:
        return float(fallback_block[-1])

    return None


def _generate_with_dp(model, **gen_kwargs):
    """
    Helper to call the generate() method on the underlying model if wrapped in DataParallel.
    """
    if hasattr(model, "module"):
        # DataParallel-wrapped: call generate() on the actual module
        return model.module.generate(**gen_kwargs)
    else:
        # Not wrapped, just call generate directly
        return model.generate(**gen_kwargs)


def evaluate_in_batch(
    model,
    tokenizer,
    device,
    test_data,
    batch_size=16,
    max_new_tokens=1200,
):
    """
    Evaluate the model in batches for faster inference.
    We'll gather all prompts, tokenize in one go,
    then slice into sub-batches to pass to `model.generate`.
    """

    # 1) Build the prompt for each test example
    prompts = []
    for item in test_data:
        question_text = item["question"]
        prompt_text = f"Question: {question_text}\nAnswer:\n"  # match training style
        prompts.append(prompt_text)

    # 2) Tokenize everything at once
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids_all = encodings["input_ids"]
    attention_mask_all = encodings["attention_mask"]
    num_samples = len(prompts)

    correct_predictions = 0
    missing_predictions = 0

    # 3) Inference in sub-batches
    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        input_ids_batch = input_ids_all[start_idx:end_idx].to(device)
        attn_mask_batch = attention_mask_all[start_idx:end_idx].to(device)

        with torch.no_grad():
            # Use our helper to handle DataParallel
            outputs = _generate_with_dp(
                model,
                input_ids=input_ids_batch,
                attention_mask=attn_mask_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 4) Decode each output in the batch
        for i, output_seq in enumerate(outputs):
            idx = start_idx + i
            generated_text = tokenizer.decode(output_seq, skip_special_tokens=False)
            predicted_answer = extract_final_answer(generated_text)

            # Compare with ground truth
            ground_truth = float(test_data[idx]["answer_value"])
            if predicted_answer is None:
                missing_predictions += 1
            elif predicted_answer == ground_truth:
                correct_predictions += 1

    total_samples = num_samples
    accuracy = correct_predictions / total_samples
    missing_rate = missing_predictions / total_samples

    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Missing Predictions: {missing_predictions} ({missing_rate * 100:.2f}%)")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy


def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer from the specified path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Decide primary device
    if torch.cuda.is_available():
        model.to("cuda:0")
        device = "cuda"
    elif torch.backends.mps.is_available():
        model.to("mps")
        device = "mps"
    else:
        model.to("cpu")
        device = "cpu"

    # Wrap in DataParallel if multiple GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Multiple GPUs detected ({num_gpus}). Using nn.DataParallel for inference.")
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    return tokenizer, model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="fine_tuned_model", help="Path to the fine-tuned model directory")
    parser.add_argument("--test_file", type=str, default="data/test_data.json", help="Path to your test JSON file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    # Load test data
    test_data = load_test_data(args.test_file)

    # Load fine-tuned model
    tokenizer, model, device = load_model_and_tokenizer(args.model_path)

    # Evaluate in batch
    accuracy = evaluate_in_batch(
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_data=test_data,
        batch_size=args.batch_size,
        max_new_tokens=1200
    )

    print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()