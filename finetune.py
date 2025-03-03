import os
import json
import time
import re
import torch
import argparse
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from torch.utils.tensorboard import SummaryWriter

# ---------------------------
# Force all GPUs in code
# ---------------------------
# In principle, you should do this BEFORE any heavy imports that check CUDA context.
# But we'll put it here for demonstration.
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    # Build a comma-separated string of GPU indices
    gpu_ids = ",".join(str(i) for i in range(num_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"Set CUDA_VISIBLE_DEVICES to use all {num_gpus} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("Only 1 or 0 GPUs are visible; no need to set CUDA_VISIBLE_DEVICES explicitly.")


# ---------------------------
# Dataset for LM Finetuning
# ---------------------------
class ChainOfThoughtDataset(Dataset):
    """
    Reads JSON lines of form:
      {
        "item_id": "...",
        "question": "...",
        "answer_cot": "... optional ...",
        "answer_value": "... optional ..."
      }

    Produces text:

    Question: <question>
    Answer:
    <cot>
      ...(chain-of-thought if available)...
    </cot>
    <answer>The answer is <...></answer>
    """

    def __init__(self, file_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for item in raw_data:
                question = item["question"]
                # Fallback to empty chain-of-thought if not present
                raw_cot = item.get("answer_cot", "")
                ans_value = item.get("answer_value", "").strip()

                # Optionally remove trailing "The answer is <NN>"
                idx = raw_cot.find("The answer is <")
                if idx != -1:
                    raw_cot = raw_cot[:idx].rstrip()

                # Build final text
                text = (
                    f"Question: {question}\n\n"
                    f"Answer:\n"
                    f"<cot>\n{raw_cot}\n</cot>\n"
                    f"<answer>The answer is <{ans_value}></answer>"
                )
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

# ---------------------------
# Generation-based Accuracy
# ---------------------------
def extract_final_answer(generated_text):
    """
    Looks for <answer> ... </answer>.
    Then for <NN> inside it (e.g. "<72>").
    If missing, fallback to last integer in entire text.
    """
    ans_block = re.search(r"<answer>(.*?)</answer>", generated_text, flags=re.DOTALL)
    if ans_block is None:
        fallback_match = re.findall(r"\d+", generated_text)
        if fallback_match:
            return float(fallback_match[-1])
        return None

    inner_text = ans_block.group(1)
    bracket_match = re.search(r"<(\d+)>", inner_text)
    if bracket_match:
        return float(bracket_match.group(1))

    # fallback
    fallback_block = re.findall(r"\d+", inner_text)
    if fallback_block:
        return float(fallback_block[-1])
    return None


@torch.no_grad()
def evaluate_generation_accuracy_in_batch(model, tokenizer, test_data_list, device, max_new_tokens=150, inference_batch_size=8):
    """
    Evaluate generation-based numeric accuracy on test_data_list in sub-batches.
    """
    # 1) Build a list of prompts
    prompts = []
    ground_truths = []
    for item in test_data_list:
        question = item["question"]
        # If "answer_value" is missing or not parseable, fallback
        ground_truth = float(item.get("answer_value", "0.0"))
        ground_truths.append(ground_truth)

        prompt = f"Question: {question}\nAnswer:\n"
        prompts.append(prompt)

    total = len(prompts)
    correct = 0
    missing = 0

    # 2) Tokenize all prompts at once
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids_all = encodings["input_ids"]
    attn_mask_all = encodings["attention_mask"]

    # 3) Inference in sub-batches
    start_idx = 0
    while start_idx < total:
        end_idx = min(start_idx + inference_batch_size, total)
        input_ids_batch = input_ids_all[start_idx:end_idx].to(device)
        attn_mask_batch = attn_mask_all[start_idx:end_idx].to(device)

        outputs = model.generate(
            input_ids=input_ids_batch,
            attention_mask=attn_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # 4) Decode and compare
        batch_size = end_idx - start_idx
        for i in range(batch_size):
            idx = start_idx + i
            gen_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            pred_ans = extract_final_answer(gen_text)
            if pred_ans is None:
                missing += 1
            elif pred_ans == ground_truths[idx]:
                correct += 1

        start_idx = end_idx

    accuracy = correct / total if total else 0.0
    missing_rate = missing / total if total else 0.0
    return accuracy, missing_rate


# ---------------------------
# Custom Callback
# ---------------------------
class GenerationAccuracyCallback(TrainerCallback):
    """
    At the end of each epoch, run generation-based numeric accuracy on a test dataset,
    then log it to TensorBoard.
    """

    def __init__(self, test_file, tokenizer, max_new_tokens=150, inference_batch_size=8):
        super().__init__()
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.inference_batch_size = inference_batch_size

        # We'll load the test data once
        with open(test_file, "r") as f:
            self.test_data = json.load(f)
        self.writer = None  # We'll create it in on_train_begin

    def on_train_begin(self, args, state, control, **kwargs):
        # Create a SummaryWriter in the same log_dir the Trainer is using
        self.writer = SummaryWriter(log_dir=args.logging_dir)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Evaluate generation-based accuracy in sub-batches
        device = next(model.parameters()).device
        model.eval()

        accuracy, missing_rate = evaluate_generation_accuracy_in_batch(
            model, self.tokenizer, self.test_data, device,
            max_new_tokens=self.max_new_tokens,
            inference_batch_size=self.inference_batch_size
        )

        print(f"[Epoch {state.epoch:.1f}] Generation Accuracy: {accuracy*100:.2f}%, Missing: {missing_rate*100:.2f}%")

        # Log to TB
        if self.writer is not None:
            self.writer.add_scalar("test/generation_accuracy", accuracy, int(state.epoch))

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.close()


# ---------------------------
# Main script
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Use 'gpt' to stick with GPT, 'gemma' for gemma-2b, or any HF model string.")
    parser.add_argument("--train_file", type=str, default="data/train_data.json")
    parser.add_argument("--test_file", type=str, default="data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max tokens for generation eval")
    parser.add_argument("--inference_batch_size", type=int, default=8, help="Batch size for generation-based accuracy")
    args = parser.parse_args()

    # -----------------------------------------------------------
    # Decide actual model name based on user request
    # -----------------------------------------------------------
    model_name = args.model_name
    if model_name.lower() == "gpt":
        # "Use the GPT that's already there." e.g. 'gpt2'
        model_name = "gpt2"
    elif model_name.lower() == "gemma":
        # "Use gemma-2b"
        model_name = "gemma-2b"

    print(f"Using model name: {model_name}")

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {
        "additional_special_tokens": ["<cot>", "</cot>", "<answer>", "</answer>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # 3) Build train dataset
    train_dataset = ChainOfThoughtDataset(args.train_file, tokenizer)

    # 4) Build an "eval_dataset" for perplexity or dev
    eval_dataset = ChainOfThoughtDataset(args.test_file, tokenizer)

    # 5) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # -----------------------------------------------------------
    # 6) TrainingArguments with multi-GPU
    # -----------------------------------------------------------
    # We'll set ddp_backend to 'nccl' if we have more than 1 GPU.
    num_gpus_now = torch.cuda.device_count()
    ddp = "nccl" if num_gpus_now > 1 else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        evaluation_strategy="no",   # We'll let the callback handle generation-based accuracy
        learning_rate=args.lr,
        weight_decay=0.01,
        report_to=["tensorboard"],  # logs training metrics
        push_to_hub=False,
        no_cuda=False,             # Ensure we allow CUDA usage
        ddp_backend=ddp            # Use NCCL if multiple GPUs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # We could omit eval_dataset if we don't want perplexity
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 7) Add our custom callback for generation-based accuracy
    gen_acc_cb = GenerationAccuracyCallback(
        test_file=args.test_file,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        inference_batch_size=args.inference_batch_size
    )
    trainer.add_callback(gen_acc_cb)

    # 8) Train
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # 9) Final model saving
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model fine-tuned and saved to {args.output_dir}")

    print("\nTraining complete. You can inspect logs in TensorBoard, including generation-based accuracy each epoch.")
    print(f"tensorboard --logdir {args.output_dir}/logs\n")
    print(f"Execution time: {(end_time - start_time)/3600} hours")


if __name__ == "__main__":
    # For gemma
    # nohup accelerate launch --num_processes=8 finetune.py --model_name gemma --epochs 3 --batch_size 6 --inference_batch_size 6 &
    main()