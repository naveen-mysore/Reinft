import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator

# --- PEFT imports ---
from peft import PeftModel

###############################################################################
# 1) Value Head wrapper for a (LoRA-augmented) base model
###############################################################################
class GPTWithValueHead(nn.Module):
    """
    A wrapper that adds a value head to any AutoModelForCausalLM
    (including google/gemma-2-2b-it, GPT-2, etc).
    """
    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        # A simple linear layer to produce a scalar value per token
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        """
        Returns:
            lm_logits: (batch, seq_len, vocab_size)
            hidden_states: (batch, seq_len, hidden_size)
            values: (batch, seq_len)  # the scalar value for each token
        """
        # Tell the base model to also return all hidden states or the last hidden state
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        # The final language-model logits
        lm_logits = outputs.logits  # shape: (B, seq_len, vocab)

        # For the value head, we can take the final hidden state:
        # Either outputs.hidden_states[-1], or an official 'last_hidden_state' if the model provides it.
        # Check which is appropriate for your model architecture.
        # For Gemma, we typically do:
        hidden_states = outputs.hidden_states[-1]  # shape: (B, seq_len, hidden_dim)

        # Pass it through our value head
        values = self.value_head(hidden_states).squeeze(-1)  # shape: (B, seq_len)

        return lm_logits, hidden_states, values

    def generate(self, **kwargs):
        """
        Inference path for text generation. We rely on the underlying
        base_model.generate(...) method.
        """
        return self.base_model.generate(**kwargs)


###############################################################################
# 2) Some utility functions for PPO
###############################################################################
def logprobs_from_logits(logits, labels):
    """
    logits: (B, seq_len, vocab)
    labels: (B, seq_len)
    returns log_probs: (B, seq_len)
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, seq_len, vocab)
    log_probs_for_labels = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
    return log_probs_for_labels


def extract_numeric_answer(text: str):
    """Returns float if found, else None. Very naive parser."""
    import re
    matches = re.findall(r"(-?\d+(\.\d+)?)", text)
    if not matches:
        return None
    return float(matches[-1][0])

def compute_reward(generated_text: str, ground_truth_str: str, eps=1e-2):
    """
    Reward=1.0 if numeric difference <= eps, else 0.
    """
    try:
        gt = float(ground_truth_str.strip())
    except:
        return 0.0
    pred = extract_numeric_answer(generated_text)
    if pred is None:
        return 0.0
    if abs(pred - gt) <= eps:
        return 1.0
    return 0.0


###############################################################################
# 3) The PPO training loop with Accelerate
###############################################################################
def ppo_training_loop(
    accelerator: Accelerator,
    policy_model: GPTWithValueHead,
    ref_model: GPTWithValueHead,
    tokenizer,
    dataset,
    num_epochs=1,
    batch_size=2,
    max_new_tokens=32,
    lr=1e-5,
    clip_param=0.2,
    vf_coef=0.5,
    kl_coef=0.01,
    device="cuda",
):
    # We'll create an optimizer for policy_model
    optimizer = optim.Adam(policy_model.parameters(), lr=lr)

    # Prepare with accelerate
    policy_model, ref_model, optimizer = accelerator.prepare(
        policy_model, ref_model, optimizer
    )

    # Freeze all reference parameters
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # If you want a scheduler, set it up. This snippet uses constant LR:
    # total_steps = (len(dataset)//(batch_size*accelerator.num_processes))*num_epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    # Let's do random shuffle each epoch. (Better approach would shard the data.)
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            print(f"\n=== PPO Epoch {epoch+1}/{num_epochs} ===")
        idxs = np.random.permutation(len(dataset))

        for i in range(0, len(dataset), batch_size):
            batch_indices = idxs[i : i+batch_size]
            if len(batch_indices) == 0:
                continue

            # 1) Build prompts
            prompts = []
            gt_values = []
            for idx_ in batch_indices:
                sample = dataset[idx_]
                q = sample["question"]
                v = sample["answer_value"]
                pmpt = f"Question: {q}\nAnswer:"
                prompts.append(pmpt)
                gt_values.append(v)

            # 2) Generate with the current (trainable) policy
            policy_model.eval()
            with torch.no_grad():
                generated_ids_list = []
                for pmpt in prompts:
                    enc = tokenizer(pmpt, return_tensors="pt").to(device)
                    out = policy_model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_k=0,
                        top_p=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated_ids_list.append(out[0])  # shape (seq_len,)

            # 3) Compute rewards
            gen_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids_list]
            rewards = [compute_reward(t, gt) for t, gt in zip(gen_texts, gt_values)]

            # 4) Re-pad everything to do forward passes
            policy_model.train()
            max_len = max(x.size(0) for x in generated_ids_list)
            B = len(generated_ids_list)
            model_input = torch.full(
                (B, max_len),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device
            )
            attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            for j, seq_ids in enumerate(generated_ids_list):
                seq_len = seq_ids.size(0)
                model_input[j, :seq_len] = seq_ids
                attention_mask[j, :seq_len] = 1

            # old values + old logprob
            with torch.no_grad():
                old_logits, _, old_values = policy_model(model_input, attention_mask)
                old_logprob = logprobs_from_logits(old_logits[:, :-1, :], model_input[:, 1:])
                ref_logits, _, _ = ref_model(model_input, attention_mask)
                ref_logprob = logprobs_from_logits(ref_logits[:, :-1, :], model_input[:, 1:])

            # place reward on the last token
            seq_lens = attention_mask.sum(dim=1)  # shape: (B,)
            adv = []
            for j in range(B):
                end_idx = seq_lens[j].item() - 2
                baseline = old_values[j, end_idx].item()
                adv_j = rewards[j] - baseline
                adv.append(adv_j)
            adv = torch.tensor(adv, dtype=torch.float, device=device)

            # approximate kl
            mask = attention_mask[:, :-1]
            kl_per_token = (old_logprob - ref_logprob) * mask
            approx_kl = kl_per_token.sum(dim=1).mean()

            # new forward
            new_logits, _, new_values = policy_model(model_input, attention_mask)
            new_logprob = logprobs_from_logits(new_logits[:, :-1, :], model_input[:, 1:])

            # ratio
            log_ratio = (new_logprob - old_logprob) * mask
            ratio = torch.exp(log_ratio)

            # build tokenwise advantage (only last token)
            tokenwise_adv = torch.zeros_like(ratio)
            for j in range(B):
                end_idx = seq_lens[j].item() - 2
                tokenwise_adv[j, end_idx] = adv[j]

            # clipped PG
            pg_loss_1 = -tokenwise_adv * ratio
            pg_loss_2 = -tokenwise_adv * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            pg_loss = torch.max(pg_loss_1, pg_loss_2).sum(dim=1).mean()

            # value loss
            new_chosen_vals = []
            for j in range(B):
                end_idx = seq_lens[j].item() - 2
                new_chosen_vals.append(new_values[j, end_idx])
            new_chosen_vals = torch.stack(new_chosen_vals, dim=0)
            returns = adv + 0.0  # ignoring discount for minimal
            vf_loss = (new_chosen_vals - returns)**2
            vf_loss = vf_loss.mean()

            # total
            loss = pg_loss + vf_coef * vf_loss + kl_coef * approx_kl
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process and i % 40 == 0:
                print(f"[E={epoch+1}] i={i} => loss={loss.item():.4f}, PG={pg_loss.item():.4f}, VF={vf_loss.item():.4f}, KL={approx_kl.item():.4f}, R={np.mean(rewards):.2f}")

    return policy_model

###############################################################################
# 4) Main entry
###############################################################################
def main():
    from accelerate import Accelerator
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True,
                        help="E.g. 'gpt2' or 'google/gemma-2-2b-it'. Should be the same base used in SFT.")
    parser.add_argument("--lora_sft_dir", type=str, required=True,
                        help="Path to your LoRA adapter from the SFT stage. This is typically the checkpoint directory.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="A JSON file with list[{'question':'...', 'answer_value':'...'}].")
    parser.add_argument("--output_dir", type=str, default="ppo_out_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the dataset
    with open(args.train_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Load base model, then apply LoRA adapter (the SFT checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, attn_implementation="eager")
    # Wrap with the SFT LoRA adapter
    base_model = PeftModel.from_pretrained(base_model, args.lora_sft_dir)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Create the policy model with a value head
    policy_model = GPTWithValueHead(base_model)

    # Also build a reference model (freeze later)
    ref_base = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, attn_implementation="eager")
    ref_model = PeftModel.from_pretrained(ref_base, args.lora_sft_dir)
    ref_model = GPTWithValueHead(ref_model)

    device = accelerator.device

    # 4) PPO
    policy_model = ppo_training_loop(
        accelerator,
        policy_model,
        ref_model,
        tokenizer,
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    # 5) Save final adapter
    #   Because we used PeftModel, we can call .base_model to get the underlying LoRA state
    #   or see if we can do policy_model.base_model.save_pretrained.
    if accelerator.is_main_process:
        # policy_model is a GPTWithValueHead. The LoRA adapter is inside policy_model.base_model
        # We can do:
        unwrapped_model = accelerator.unwrap_model(policy_model)
        # unwrapped_model.base_model.save_pretrained(args.output_dir)  # This might save the entire HF model, plus LoRA
        # But typically for LoRA we do:
        unwrapped_model.base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print(f"Done! Final LoRA PPO model saved to {args.output_dir}")

if __name__ == "__main__":
    main()