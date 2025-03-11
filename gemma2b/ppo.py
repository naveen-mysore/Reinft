#!/usr/bin/env python3

"""
Minimal PPO script with a separate reference model for KL penalty (ReFT style),
plus logging of the average rollout rewards, and a --data_fraction argument.

Modifications to:
  - Load "prompt" and "answer_value" from the dataset.
  - Build a prompt for the policy model that mimics the reference inference style.
  - Extract numeric carbs from the model's "Output: {\"total_carbohydrates\": X}" line.
  - Compare with the gold answer_value for the reward.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7,8,9"
import json
import math
import random
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from peft import PeftModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_constant_schedule_with_warmup
)
from accelerate import Accelerator

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###############################################################################
# HELPER: parse <cot> and <answer>
###############################################################################

accelerator = Accelerator()
llm_cot_prompt_gemma2 = """<bos><start_of_turn>user
For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size.
   If the serving size of any item is not specified, assume it is a single standard serving based on
   common nutritional guidelines (e.g., USDA). Ignore additional information that doesn’t relate to
   the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for
   the specific serving size.
3. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{{"total_carbohydrates": total grams of carbohydrates for the serving}}

For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text.
If you don’t know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering:

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer:
<cot>
Let’s think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana, and 1 glass of orange juice.
1 cup of oatmeal has 27g carbs.
1 banana has 27g carbs, so half a banana has (27 * 1/2) = 13.5g carbs.
1 glass of orange juice has 26g carbs.
So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5
Output: {{"total_carbohydrates": 66.5}}
</cot>
<answer>The answer is <66.5></answer>
Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer:
<cot>
Let’s think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs with 2 eggs has 2g carbs.
1 toast has 13g carbs.
So the total grams of carbs in the meal = (2 + 13) = 15
Output: {{"total_carbohydrates": 15}}
</cot>
<answer>The answer is <15></answer>

Query: "Half a peanut butter and jelly sandwich."
Answer:
<cot>
Let’s think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 50.6g carbs, so half a sandwich is (50.6 * 1/2) = 25.3g carbs.
So the total grams of carbs in the meal = 25.3
Output: {{"total_carbohydrates": 25.3}}
</cot>
<answer>The answer is <25.3></answer>
Query: {query}
Answer:
<cot>
"""
def parse_cot_and_answer(generated_text, user_prompt):
    # (Unused in the main code, but left here if needed)
    half_pb_idx = generated_text.find("Half a peanut butter")
    if half_pb_idx == -1:
        return "", ""

    start_idx = generated_text.find("Query:", half_pb_idx)
    if start_idx == -1:
        return "", ""

    end_idx = generated_text.find("</answer>", start_idx)
    if end_idx == -1:
        return "", ""

    text_slice = generated_text[start_idx : end_idx + len("</answer>")]
    match_cot = re.search(r"<cot>(.*?)</cot>", text_slice, flags=re.DOTALL)
    chain_of_thought = match_cot.group(1).strip() if match_cot else ""

    match_ans = re.search(r"<answer>(.*?)</answer>", text_slice, flags=re.DOTALL)
    final_answer = match_ans.group(1).strip() if match_ans else ""

    return chain_of_thought, final_answer

def extract_carbs_from_answer_block(answer_text: str) -> Optional[float]:
    # e.g. final answer might look like: "The answer is <26.8>"
    match = re.search(r"<([\d.]+)>", answer_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None

def compare_answer(pred_value: float, true_value: float, margin=7.0):
    accelerator.print(f"{pred_value} vs {true_value}")
    return abs(pred_value - true_value) <= margin

def compute_logprobs_from_logits(logits, labels):
    """
    logits: [B, seq_len, vocab]
    labels: [B, seq_len]
    returns logprobs: [B, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    labels_flat = labels.unsqueeze(-1)  # [B, seq_len, 1]
    token_logprobs = torch.gather(log_probs, dim=-1, index=labels_flat).squeeze(-1)
    return token_logprobs

###############################################################################
# 2) POLICY + VALUE HEAD
###############################################################################
class PolicyValueModel(nn.Module):
    """
    Wraps a base model (AutoModelForCausalLM),
    and adds a small linear layer as the "value head" -> [B, seq_len].
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        # out.logits: [B, seq_len, vocab]
        # out.hidden_states[-1]: [B, seq_len, hidden_dim]
        last_hidden = out.hidden_states[-1]
        lm_logits = out.logits
        values = self.v_head(last_hidden).squeeze(-1)  # [B, seq_len]
        return lm_logits, values

###############################################################################
# 4) ROLLOUT
###############################################################################
def rollout_step(
    model: PolicyValueModel,
    tokenizer,
    prompts,
    true_values,
    max_new_tokens=200,
):
    device = next(model.parameters()).device

    # 1) Tokenize the prompt
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    batch_size = input_ids.size(0)

    # 2) Record the length of the non-pad tokens in the prompt
    prompt_lens = attention_mask.sum(dim=1)  # shape [batch]

    # 3) Generate
    model.eval()
    with torch.no_grad():
        generation = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    model.train()

    # 4) Compute reward for each sample
    completions = tokenizer.batch_decode(generation, skip_special_tokens=False)
    reward_list = []
    for i, gen_text in enumerate(completions):
        chain_of_thought, final_answer = parse_cot_and_answer(gen_text, prompts[i])
        guess = extract_carbs_from_answer_block(final_answer)
        if guess is not None and compare_answer(guess, true_values[i]):
            reward_list.append(1.0)
        else:
            reward_list.append(0.0)

    # 5) Re-run the entire sequence (prompt+generated tokens) to get old logprobs/values
    model_input_ids = generation
    att_mask = (model_input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        lm_logits, values = model(model_input_ids, att_mask)
        # Now compute the logprobs for next-token predictions
        logprobs = compute_logprobs_from_logits(
            lm_logits[:, :-1, :],
            model_input_ids[:, 1:]
        )
        # Align shapes
        pad_col = torch.zeros(size=(batch_size, 1), device=logprobs.device)
        old_logprobs = torch.cat([logprobs, pad_col], dim=1)
        old_values   = values

    # 6) Place the final reward at the last *non-pad* token
    reward_tensor = torch.zeros_like(old_values)
    for i in range(batch_size):
        seq_len_i = att_mask[i].sum().item()
        reward_tensor[i, seq_len_i - 1] = reward_list[i]

    # 7) Build a train_mask to exclude the “echoed prompt” portion
    train_mask = torch.zeros_like(model_input_ids, dtype=torch.float, device=device)
    for i in range(batch_size):
        seq_len_i = att_mask[i].sum().item()
        start = prompt_lens[i].item()
        # mark 1.0 for chain-of-thought portion
        train_mask[i, start:seq_len_i] = 1.0

    return model_input_ids, att_mask, old_logprobs, old_values, reward_tensor, train_mask, reward_list

###############################################################################
# 5) PPO STEP with REF MODEL (But not actually used for KL as-is!)
###############################################################################
def ppo_step(
    model: PolicyValueModel,
    optimizer,
    old_logprobs,
    old_values,
    model_input_ids,
    attention_mask,
    reward_tensor,
    train_mask,
    # Additional args for your kl
    ref_model,
    kl_coef=0.02,
    clip_range=0.2,
    vf_coef=1.0,
    gamma=1.0,
    lam=0.95,
):
    """
    Perform one PPO update step, plus an extra KL penalty that compares
    the current policy to a frozen reference model.
    """
    # ------------------------------------------------------------------
    # 1) Forward pass on the *current* policy to get new logprobs/values
    # ------------------------------------------------------------------
    lm_logits, new_values = model(model_input_ids, attention_mask)

    # Next-token logprobs
    logprobs = compute_logprobs_from_logits(
        lm_logits[:, :-1, :],
        model_input_ids[:, 1:]
    )
    pad_col = torch.zeros(size=(logprobs.size(0), 1), device=logprobs.device)
    new_logprobs = torch.cat([logprobs, pad_col], dim=1)

    valid_mask = attention_mask.float() * train_mask

    # ------------------------------------------------------------------
    # 2) Compute advantages via GAE, using old_values for baseline
    # ------------------------------------------------------------------
    seq_len = new_values.size(1)
    advantages = torch.zeros_like(reward_tensor)
    last_gae = 0.0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_val = 0.0
        else:
            next_val = new_values[:, t+1]

        delta = reward_tensor[:, t] + gamma * next_val - old_values[:, t]
        advantages[:, t] = last_gae = delta + gamma * lam * last_gae

    returns = advantages + old_values

    # ------------------------------------------------------------------
    # 3) Standard PPO policy loss (clip objective)
    # ------------------------------------------------------------------
    ratio = (new_logprobs - old_logprobs).exp()  # pi_new / pi_old
    ratio_masked = ratio * valid_mask
    adv_masked   = advantages * valid_mask

    pg_loss1 = -adv_masked * ratio_masked
    pg_loss2 = -adv_masked * torch.clamp(ratio_masked, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss  = torch.max(pg_loss1, pg_loss2)

    mask_sum = valid_mask.sum()
    policy_loss = pg_loss.sum() / mask_sum.clamp_min(1.0)

    # ------------------------------------------------------------------
    # 4) Value function loss (clip objective)
    # ------------------------------------------------------------------
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
    vf_loss1 = (new_values - returns) ** 2
    vf_loss2 = (v_clipped - returns) ** 2
    vf_loss  = 0.5 * torch.max(vf_loss1, vf_loss2)
    value_loss = (vf_loss * valid_mask).sum() / mask_sum.clamp_min(1.0)

    # ------------------------------------------------------------------
    # 5) KL penalty vs reference model
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Correct reference model forward pass
        ref_out = ref_model(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        ref_logits = ref_out.logits

        # Compute reference logprobs
        ref_logprobs_ = compute_logprobs_from_logits(
            ref_logits[:, :-1, :],
            model_input_ids[:, 1:]
        )
        ref_logprobs_ = torch.cat([ref_logprobs_, pad_col], dim=1)

    # Ratio between new policy and ref policy
    log_ratio_ref = (new_logprobs - ref_logprobs_)
    ratio_ref = log_ratio_ref.exp()  # pi_new / pi_ref

    # Forward KL approx: E_{x ~ pi_new}[ log(pi_new / pi_ref) ]
    kl_per_token = ratio_ref * log_ratio_ref

    # Only apply to the valid portion
    kl_per_token = kl_per_token * valid_mask
    kl_mean = kl_per_token.sum() / mask_sum.clamp_min(1.0)

    # Weighted penalty
    kl_loss = kl_coef * kl_mean

    # ------------------------------------------------------------------
    # 6) Final combined loss
    # ------------------------------------------------------------------
    total_loss = policy_loss + vf_coef * value_loss + kl_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_value": kl_mean.item(),   # log how big the KL is
        "kl_loss": kl_loss.item(),    # log the actual added penalty
        "total_loss": total_loss.item(),
    }

###############################################################################
# 6) MAIN
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True,
                        help="Base HF model name or path, e.g. 'google/gemma-2-2b-it'")
    parser.add_argument("--peft_checkpoint", type=str, default=None, required=True,
                        help="Path to LoRA/PEFT adapter (warm-up checkpoint).")
    parser.add_argument("--train_file", type=str, required=True,
                        help="JSON data with { 'prompt': ..., 'answer_value': ... }.")
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed + accelerator.process_index)

    # --------------------------------------------------------------------------
    # A) Load data
    # --------------------------------------------------------------------------
    with open(args.train_file, "r") as f:
        data = json.load(f)

    # Expect items: { "prompt": ..., "answer_value": ... }
    dataset = []
    for item in data:
        q = item["prompt"]
        ans = float(item["answer_value"])
        dataset.append((q, ans))

    if args.data_fraction < 1.0:
        random.shuffle(dataset)
        keep_size = int(len(dataset) * args.data_fraction)
        dataset = dataset[:keep_size]

    accelerator.print(f"Using {len(dataset)} data items (fraction={args.data_fraction}).")

    # --------------------------------------------------------------------------
    # B) Load base model (trainable) + reference model (frozen)
    # --------------------------------------------------------------------------
    accelerator.print(f"Loading base model {args.base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto"  # Might conflict w/ accelerate on some setups
    )
    if args.peft_checkpoint:
        accelerator.print(f"Loading PEFT from {args.peft_checkpoint}")
        base_model = PeftModel.from_pretrained(base_model, args.peft_checkpoint)

    # Create a separate reference model from the same checkpoint or base
    accelerator.print("Loading reference model (frozen)...")
    ref_base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto"
    )
    if args.peft_checkpoint:
        ref_base_model = PeftModel.from_pretrained(ref_base_model, args.peft_checkpoint)

    # Freeze the reference model's parameters
    for param in ref_base_model.parameters():
        param.requires_grad = False

    policy_value_model = PolicyValueModel(base_model)

    # --------------------------------------------------------------------------
    # C) Build tokenizer
    # --------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 2
    tokenizer.padding_side = "left"

    # Prepare for distribution
    policy_value_model, ref_base_model = accelerator.prepare(policy_value_model, ref_base_model)

    # --------------------------------------------------------------------------
    # D) Optimizer
    # --------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(policy_value_model.parameters(), lr=args.lr)
    schedule = get_constant_schedule_with_warmup(optimizer, 0)

    global_step = 0
    for epoch in range(args.n_epochs):
        random.shuffle(dataset)
        steps_per_epoch = math.ceil(len(dataset) / args.batch_size)

        for step_i in range(steps_per_epoch):
            batch_slice = dataset[step_i * args.batch_size : (step_i + 1) * args.batch_size]
            if not batch_slice:
                break

            batch_q = [x[0] for x in batch_slice]
            batch_a = [x[1] for x in batch_slice]

            # Build prompts
            prompts = [llm_cot_prompt_gemma2.format(query=q) for q in batch_q]

            # Rollout
            (model_input_ids,
             attention_mask,
             old_logprobs,
             old_values,
             reward_tensor,
             train_mask,
             reward_list) = rollout_step(
                policy_value_model,
                tokenizer,
                prompts,
                batch_a,
            )
            avg_reward = sum(reward_list) / len(reward_list)

            # PPO updates
            for _ in range(args.ppo_epochs):
                stats = ppo_step(
                    policy_value_model,
                    optimizer,
                    old_logprobs,
                    old_values,
                    model_input_ids,
                    attention_mask,
                    reward_tensor,
                    train_mask,
                    ref_model=ref_base_model,
                    kl_coef=args.kl_coef,
                    clip_range=0.2,
                    vf_coef=1.0,
                    gamma=1.0,
                    lam=0.95,
                )

            schedule.step()
            global_step += 1

            # Save model checkpoint periodically
            if global_step % 10 == 0:
                accelerator.print(f"Saving model checkpoint at step {global_step}")
                unwrapped_policy_value = accelerator.unwrap_model(policy_value_model)
                checkpoint_dir = f"ppo_out/checkpoint_{global_step}"

                unwrapped_policy_value.base_model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

            if global_step % 1 == 0:
                accelerator.print(
                    f"Epoch={epoch}, step={global_step}, "
                    f"avg_reward={avg_reward:.3f}, stats={stats}"
                )

    accelerator.print("Done PPO training!")
    accelerator.wait_for_everyone()

    # --------------------------------------------------------------------------
    # E) Save the final model
    # --------------------------------------------------------------------------
    unwrapped_policy_value = accelerator.unwrap_model(policy_value_model)

    # 1) Save *only* the base model (for normal inference)
    unwrapped_policy_value.base_model.save_pretrained("ppo_trained_model")
    tokenizer.save_pretrained("ppo_trained_model")

    # 2) (Optional) Save the entire policy+value model
    torch.save(unwrapped_policy_value.state_dict(), "ppo_policy_value_model.pt")


if __name__ == "__main__":
    main()