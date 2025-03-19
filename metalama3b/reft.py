#!/usr/bin/env python3
"""
ReFT PPO script for a LLaMA-based model (text-only),
using Option 2: Keep a separate reference to the base LM unwrapped,
so we can call raw_lm.generate(...) even after DDP wrapping policy_value_model.
"""

import os
import json
import math
import random
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from collections import deque
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    get_constant_schedule_with_warmup
)

from prompt_manager import PromptManager

###############################################################################
# 1) PPO REWARD + ADVANTAGE UTILS
###############################################################################
def compute_logprobs_from_logits(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    labels_flat = labels.unsqueeze(-1)
    token_logprobs = torch.gather(log_probs, dim=-1, index=labels_flat).squeeze(-1)
    return token_logprobs


def gaussian_reward(
        pred_value: float,
        true_value: float,
        sigma: float = 10.0,
        near_exact_range: float = 1.0,
        near_exact_bonus: float = 1.0
):
    """
    Computes a reward in (0, +∞) based on a Gaussian decay from the ground truth.

    1. The base reward is exp( - (diff^2) / (2*sigma^2) ),
       where diff = (pred_value - true_value).
       - If diff=0, reward=1.0.
       - For large |diff|, reward approaches 0.0.
    2. If the guess is within 'near_exact_range', we add a
       'near_exact_bonus' to the result, allowing the final
       reward to exceed 1.0.

    Example:
        >>> # 1) If pred=60, truth=60 => diff=0 => base=1.0, final=1.0+bonus
        >>> # 2) If pred=65 => diff=5 => base=exp(-25/200)=exp(-0.125)≈0.88
        >>> # 3) If pred=80 => diff=20 => base=exp(-400/200)=exp(-2)≈0.135
    """
    diff = abs(pred_value - true_value)
    # Base Gaussian
    base_reward = math.exp(-(diff ** 2) / (2.0 * sigma * sigma))

    # near-exact bonus region
    if diff <= near_exact_range:
        base_reward += near_exact_bonus

    return base_reward


def scaled_reward(pred_value: float, true_value: float, threshold=5.0, accelerator=None):
    """
    A simple numeric reward function for how close 'pred_value' is to 'true_value',
    giving up to 1.0 if within 'threshold' range, else 0.0.
    """
    diff = abs(pred_value - true_value)
    if diff >= threshold:
        return 0.0
    else:
        # linear scale from 1.0 down to 0.0 as diff goes 0..threshold
        return 1.0 - (diff / threshold)


def compute_gae_advantages(rewards, old_values, gamma=0.95, lam=0.95):
    seq_len = old_values.size(1)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_val = 0.0
        else:
            next_val = old_values[:, t + 1]
        delta = rewards[:, t] + gamma * next_val - old_values[:, t]
        advantages[:, t] = last_gae = delta + gamma * lam * last_gae
    returns_ = advantages + old_values
    return advantages, returns_

###############################################################################
# 2) POLICY-VALUE MODEL WRAPPER
###############################################################################
class PolicyValueModel(nn.Module):
    """
    Wraps a LlamaForCausalLM + an extra value head in text-only mode.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size  # LLaMA uses `hidden_size`
        # Minimal value head
        self.v_head = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        lm_logits = out.logits  # [B, seq_len, vocab]
        last_hidden = out.hidden_states[-1]  # [B, seq_len, hidden_size]
        values = self.v_head(last_hidden).squeeze(-1)  # [B, seq_len]
        return lm_logits, values


###############################################################################
# 3) ROLLOUT STEP
###############################################################################
def find_sublist(sub, main):
    """
    Returns the starting index of list 'sub' in list 'main', or -1 if not found.
    """
    for i in range(len(main) - len(sub) + 1):
        if main[i:i + len(sub)] == sub:
            return i
    return -1


def rollout_step(
    policy_value_model: PolicyValueModel,
    raw_lm,
    tokenizer,
    prompts,
    true_values,
    prompt_manager,
    max_new_tokens=200,
    accelerator=None,
    temperature=1.0,
    do_sample=False
):
    """
    Generate text with the same prompt style as inference (so <cot> and <answer> appear),
    then parse and compute rewards. Use raw_lm for .generate() to avoid DDP issues.
    """
    device = next(policy_value_model.parameters()).device
    batch_size = len(prompts)

    # Build training prompts
    training_prompts = [prompt_manager.build_prompt(p, mode="training") for p in prompts]
    enc = tokenizer(training_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 1) Generate final sequences with raw_lm
    raw_lm.eval()
    with torch.no_grad():
        outputs = raw_lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    raw_lm.train()

    final_texts = [
        tokenizer.decode(seq, skip_special_tokens=False)
        for seq in outputs
    ]

    # 2) Parse <cot> and <answer>, compute reward VALUES (without assigning yet)
    reward_list = []
    pred_values = []

    for i, text in enumerate(final_texts):
        has_cot = ("<cot>" in text) and ("</cot>" in text)
        has_answer = ("<answer>" in text) and ("</answer>" in text)
        has_output_dict = False
        if has_answer and "Output: {\"total_carbohydrates\":" in text:
            has_output_dict = True
        correct_structure = has_cot and has_answer and has_output_dict
        structure_reward = 0.02 if correct_structure else 0.0

        chain_of_thought, final_ans = prompt_manager.parse_cot_and_answer(
            text,
            verbose=accelerator is not None,
            log_fn=accelerator.print if accelerator else None
        )
        guess = prompt_manager.extract_carbs_from_answer(final_ans)
        pred_values.append(guess)

        # Check if the answer is malformed
        is_malformed = False
        if final_ans:
            # We can try a quick format check:
            clean_json = re.match(r'.*?"total_carbohydrates":\s*"?(\d+\.?\d*)"?\s*}.*?', final_ans)
            clean_output = re.match(r'Output:\s*(\d+\.?\d*)\s*$', final_ans)
            json_with_garbage = re.search(r'"total_carbohydrates":\s*"?(\d+\.?\d*)[^"}]*[^0-9\.\s"}]', final_ans)
            if (not (clean_json or clean_output)) or json_with_garbage:
                is_malformed = True

        if guess is None or is_malformed:
            numeric_reward = 0.0
        else:
            numeric_reward = scaled_reward(guess, true_values[i], threshold=20.0, accelerator=accelerator)
            # or: numeric_reward = gaussian_reward(guess, true_values[i], sigma=5.0)

        reward_val = numeric_reward + structure_reward
        reward_list.append(reward_val)

    # 3) Re-encode the full texts to get the full token sequences
    enc_full = tokenizer(final_texts, return_tensors="pt", padding=True, truncation=True)
    full_input_ids = enc_full["input_ids"].to(device)
    full_attention_mask = enc_full["attention_mask"].to(device)

    with torch.no_grad():
        lm_logits_full, values_full = policy_value_model(full_input_ids, full_attention_mask)

    old_logprobs = compute_logprobs_from_logits(lm_logits_full[:, :-1, :], full_input_ids[:, 1:])
    padcol = torch.zeros((old_logprobs.size(0), 1), device=device)
    old_logprobs = torch.cat([old_logprobs, padcol], dim=1)

    seq_lens = full_attention_mask.sum(dim=1)
    reward_tensor = torch.zeros_like(values_full)

    # 4) Assign rewards to the tokens containing the numeric guess
    reward_token_info = []
    for b in range(batch_size):
        last_idx = seq_lens[b].item() - 1
        final_reward = reward_list[b]
        guess = pred_values[b]

        reward_assignment = {"token_indices": [], "token_text": ""}

        if guess is not None:
            guess_str = str(guess)
            candidates = [guess_str]
            if '.' in guess_str:
                if guess_str.endswith('.0'):
                    candidates.append(guess_str[:-2])
                base, decimal = guess_str.split('.', 1)
                if len(decimal) > 1:
                    candidates.append(f"{base}.{decimal[:-1]}")

            found = False
            for candidate in candidates:
                guess_tokens = tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if not guess_tokens:
                    continue

                full_ids_list = full_input_ids[b].tolist()
                found_idx = find_sublist(guess_tokens, full_ids_list)
                if found_idx != -1:
                    assigned_indices = []
                    for j in range(len(guess_tokens)):
                        reward_tensor[b, found_idx + j] = final_reward
                        assigned_indices.append(found_idx + j)
                    reward_assignment["token_indices"] = assigned_indices
                    reward_assignment["token_text"] = candidate
                    found = True
                    break

            if not found:
                reward_tensor[b, last_idx] = final_reward
                reward_assignment["token_indices"] = [last_idx]
                reward_assignment["token_text"] = "last token (fallback)"
        else:
            reward_tensor[b, last_idx] = final_reward
            reward_assignment["token_indices"] = [last_idx]
            reward_assignment["token_text"] = "last token (no prediction)"

        reward_token_info.append(reward_assignment)

    # 5) Build train_mask to only include newly generated tokens
    train_mask = torch.zeros_like(values_full)
    for b in range(batch_size):
        original_prompt_len = input_ids[b].shape[0]
        if original_prompt_len < seq_lens[b]:
            train_mask[b, original_prompt_len:seq_lens[b]] = 1.0

    # 6) Create CoT mask to up-weight CoT tokens in KL calculation
    cot_mask = torch.zeros_like(values_full)
    for b in range(batch_size):
        text = final_texts[b]
        if "<cot>" in text and "</cot>" in text:
            cot_start = text.find("<cot>") + len("<cot>")
            cot_end = text.find("</cot>")
            if cot_start < cot_end:
                cot_text = text[cot_start:cot_end]
                cot_tokens = tokenizer(cot_text, add_special_tokens=False)["input_ids"]
                full_ids = full_input_ids[b].tolist()
                cot_start_idx = find_sublist(cot_tokens, full_ids)
                if cot_start_idx != -1:
                    for j in range(len(cot_tokens)):
                        cot_mask[b, cot_start_idx + j] = 1.0

    return (
        full_input_ids,
        full_attention_mask,
        old_logprobs,
        values_full,
        reward_tensor,
        train_mask,
        reward_list,
        cot_mask,
        final_texts,
        reward_token_info
    )

###############################################################################
# 4) PPO STEP
###############################################################################
def ppo_step(
    model: PolicyValueModel,
    optimizer,
    old_logprobs,
    old_values,
    input_ids,
    attn_mask,
    reward_tensor,
    train_mask,
    ref_model,
    advantages,
    returns,
    cot_mask,
    kl_coef=0.02,
    clip_range=0.2,
    vf_coef=1.0,
    cot_kl_discount=0.5,
    entropy_coef=0.0
):
    """
    Run a PPO update on a batch of trajectories, with optional entropy bonus.
    """
    lm_logits, new_values = model(input_ids, attn_mask)
    logprobs = compute_logprobs_from_logits(lm_logits[:, :-1, :], input_ids[:, 1:])
    batch_sz = logprobs.size(0)
    padcol = torch.zeros((batch_sz, 1), device=logprobs.device, dtype=logprobs.dtype)
    new_logprobs = torch.cat([logprobs, padcol], dim=1)

    valid_mask = attn_mask.float() * train_mask
    ratio = (new_logprobs - old_logprobs).exp()
    ratio_masked = ratio * valid_mask
    adv_masked = advantages * valid_mask

    pg_loss1 = -adv_masked * ratio_masked
    pg_loss2 = -adv_masked * torch.clamp(ratio_masked, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss_ = torch.max(pg_loss1, pg_loss2)
    mask_sum = valid_mask.sum().clamp_min(1.0)
    policy_loss = pg_loss_.sum() / mask_sum

    # Value Loss
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
    vf_loss1 = (new_values - returns) ** 2
    vf_loss2 = (v_clipped - returns) ** 2
    vf_loss_ = 0.5 * torch.max(vf_loss1, vf_loss2)
    value_loss = (vf_loss_ * valid_mask).sum() / mask_sum

    # KL vs reference model with CoT discount
    with torch.no_grad():
        ref_out = ref_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
            use_cache=False
        )
        ref_logits = ref_out.logits
        rlogprobs = compute_logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
        rlogprobs = torch.cat([rlogprobs, padcol], dim=1)

    log_ratio = new_logprobs - rlogprobs
    ratio_ref = log_ratio.exp()
    kl_per_token = ratio_ref * log_ratio

    # Apply KL discount to CoT tokens
    kl_scaling = (1.0 - cot_mask) + (cot_kl_discount * cot_mask)
    kl_per_token = kl_per_token * valid_mask * kl_scaling
    scaled_mask_sum = (valid_mask * kl_scaling).sum().clamp_min(1.0)
    kl_mean = kl_per_token.sum() / scaled_mask_sum
    kl_loss = kl_coef * kl_mean

    # Entropy bonus
    entropy_loss = 0.0
    mean_entropy = 0.0
    if entropy_coef > 0:
        with torch.no_grad():
            dist = F.softmax(lm_logits[:, :-1, :], dim=-1)
            log_dist = F.log_softmax(lm_logits[:, :-1, :], dim=-1)
            entropy_per_token = -(dist * log_dist).sum(dim=-1)
            entropy_pad = torch.zeros((batch_sz, 1), device=entropy_per_token.device,
                                      dtype=entropy_per_token.dtype)
            entropy_per_token_full = torch.cat([entropy_per_token, entropy_pad], dim=1)
            masked_entropy = entropy_per_token_full * valid_mask
            mean_entropy = masked_entropy.sum() / mask_sum
        entropy_loss = -entropy_coef * mean_entropy

    # Total loss
    total_loss = policy_loss + vf_coef * value_loss + kl_loss + entropy_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_value": kl_mean.item(),
        "kl_loss": kl_loss.item(),
        "entropy": mean_entropy.item(),
        "entropy_loss": entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(),
        "total_loss": total_loss.item()
    }


###############################################################################
# 5) MAIN
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm_start_model", type=str, default="fine_tuned_model",
                        help="Path to your *fine-tuned* model checkpoint.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--kl_coef", type=float, default=0.0001)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--data_fraction", type=float, default=0.6)
    parser.add_argument("--wandb_project", type=str, default="ppo_nutri_g3")
    parser.add_argument("--wandb_entity", type=str, default="nmysore-uc-santa-barbara")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--do_sample", action="store_true", default="True", help="Use sampling instead of greedy.")
    parser.add_argument("--ppo_epochs", type=int, default=2,
                        help="Number of PPO update epochs per batch")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range parameter")
    parser.add_argument("--cot_kl_discount", type=float, default=0.5,
                        help="Discount factor for KL penalty on CoT tokens (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, default="ppo_trained_model",
                        help="Where to store the final PPO model (so inference can load it).")
    parser.add_argument("--entropy_coef", type=float, default=0.1,
                        help="Coefficient for entropy bonus in PPO (0.0 to disable)")
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # set seed
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + accelerator.process_index)

    if accelerator.is_main_process and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )

    # 1) Load data
    with open(args.train_file, "r") as f:
        data = json.load(f)
    dataset = []
    for item in data:
        q = item["question"]
        val = float(item["answer_value"])
        dataset.append((q, val))
    if args.data_fraction < 1.0:
        random.shuffle(dataset)
        keep_sz = int(len(dataset) * args.data_fraction)
        dataset = dataset[:keep_sz]

    accelerator.print(f"Loaded {len(dataset)} training samples from {args.train_file}")

    # 2) Load raw model for generation
    accelerator.print(f"Loading raw_lm from {args.warm_start_model}")
    config = AutoConfig.from_pretrained(args.warm_start_model)
    config.use_cache = False

    raw_lm = LlamaForCausalLM.from_pretrained(
        args.warm_start_model,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )

    # reference model
    accelerator.print("Loading reference model from the same checkpoint")
    ref_base = LlamaForCausalLM.from_pretrained(
        args.warm_start_model,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    for p in ref_base.parameters():
        p.requires_grad = False

    # 3) Build policy-value model
    policy_value_model = PolicyValueModel(raw_lm)
    policy_value_model.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

    # 4) Load tokenizer
    accelerator.print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.warm_start_model)
    # For LLaMA-based tokenizers, you may need: use_fast=False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Wrap for DDP
    policy_value_model, ref_base = accelerator.prepare(policy_value_model, ref_base)

    # 5) optimizer
    optimizer = torch.optim.AdamW(policy_value_model.parameters(), lr=args.lr)
    sched = get_constant_schedule_with_warmup(optimizer, 0)

    global_step = 0
    reward_deque = deque(maxlen=100)

    # Initialize PromptManager
    prompt_manager = PromptManager()

    # 6) PPO
    for ep in range(args.n_epochs):
        random.shuffle(dataset)
        steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
        for step_i in range(steps_per_epoch):
            batch_slice = dataset[step_i * args.batch_size:(step_i + 1) * args.batch_size]
            if not batch_slice:
                break
            b_q = [x[0] for x in batch_slice]
            b_a = [x[1] for x in batch_slice]

            # Rollout
            (
                new_input_ids,
                new_attn_mask,
                old_logprobs,
                old_values,
                reward_tensor,
                train_mask,
                reward_list,
                cot_mask,
                final_texts,
                reward_token_info
            ) = rollout_step(
                policy_value_model,
                raw_lm,
                tokenizer,
                b_q,
                b_a,
                prompt_manager,
                max_new_tokens=args.max_new_tokens,
                accelerator=accelerator,
                temperature=args.temperature,
                do_sample=args.do_sample
            )

            avg_reward = sum(reward_list) / len(reward_list)
            reward_deque.append(avg_reward)
            rolling_avg = sum(reward_deque) / len(reward_deque)

            advantages, returns_ = compute_gae_advantages(reward_tensor, old_values)

            # Add advantage whitening
            with torch.no_grad():
                advantages = advantages * train_mask
                valid_mask_bool = (train_mask > 0).bool()
                valid_adv = advantages[valid_mask_bool]
                mean_adv = valid_adv.mean()
                std_adv = valid_adv.std() + 1e-8
                advantages = (advantages - mean_adv) / std_adv

            # Parse results for logging
            pred_values = []
            parsed_cot = []
            parsed_answer = []
            for i, gen_text in enumerate(final_texts):
                chain_of_thought, final_ans = prompt_manager.parse_cot_and_answer(gen_text)
                guess = prompt_manager.extract_carbs_from_answer(final_ans)
                pred_values.append(guess)
                parsed_cot.append(chain_of_thought)
                parsed_answer.append(final_ans)

            # Debug prints
            if accelerator and accelerator.is_main_process:
                accelerator.print("\n----- Batch Predictions -----")
                for i in range(len(b_q)):
                    accelerator.print(f"Example {i+1}:")
                    query_trunc = b_q[i][:50] + "..." if len(b_q[i]) > 50 else b_q[i]
                    accelerator.print(f"  Query: {query_trunc}")

                    cot_trunc = parsed_cot[i][:100] + "..." if len(parsed_cot[i]) > 100 else parsed_cot[i]
                    accelerator.print(f"  Generated CoT: {cot_trunc}")
                    accelerator.print(f"  Generated Answer: {parsed_answer[i]}")
                    accelerator.print(f"  Pred={pred_values[i]}")
                    accelerator.print(f"  True={b_a[i]}")
                    accelerator.print(f"  Reward={reward_list[i]}")
                    token_text = reward_token_info[i]["token_text"]
                    token_indices = reward_token_info[i]["token_indices"]
                    accelerator.print(f"  Reward assigned to: '{token_text}' at indices {token_indices}")
                    accelerator.print("----------------------------")

            # Multiple PPO epochs
            last_loss_dict = None
            for ppo_epoch in range(args.ppo_epochs):
                loss_dict = ppo_step(
                    model=policy_value_model,
                    optimizer=optimizer,
                    old_logprobs=old_logprobs,
                    old_values=old_values,
                    input_ids=new_input_ids,
                    attn_mask=new_attn_mask,
                    reward_tensor=reward_tensor,
                    train_mask=train_mask,
                    ref_model=ref_base,
                    advantages=advantages,
                    returns=returns_,
                    cot_mask=cot_mask,
                    kl_coef=args.kl_coef,
                    clip_range=args.clip_range,
                    vf_coef=1.0,
                    cot_kl_discount=args.cot_kl_discount,
                    entropy_coef=args.entropy_coef
                )
                sched.step()
                last_loss_dict = loss_dict
            global_step += 1

            accelerator.wait_for_everyone()

            # Logging
            if accelerator.is_main_process and wandb.run and last_loss_dict is not None:
                wandb.log({
                    "train/epoch": ep,
                    "train/step": global_step,
                    "train/avg_reward": avg_reward,
                    "train/rolling_avg_reward": rolling_avg,
                    "train/policy_loss": last_loss_dict["policy_loss"],
                    "train/value_loss": last_loss_dict["value_loss"],
                    "train/kl_value": last_loss_dict["kl_value"],
                    "train/kl_loss": last_loss_dict["kl_loss"],
                    "train/entropy": last_loss_dict["entropy"],
                    "train/entropy_loss": last_loss_dict["entropy_loss"],
                    "train/total_loss": last_loss_dict["total_loss"]
                }, step=global_step)

            # Optional checkpoint saving
            if global_step % 200 == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.print(f"Saving checkpoint at step={global_step}")
                    unwrap = accelerator.unwrap_model(policy_value_model)
                    ckdir = f"ppo_out/checkpoint_{global_step}"
                    unwrap.base_model.save_pretrained(ckdir)
                    tokenizer.save_pretrained(ckdir)
                accelerator.wait_for_everyone()

        accelerator.print(f"Epoch={ep} completed. Rolling avg reward={rolling_avg:.3f}")

    accelerator.print("Done PPO training!")
    accelerator.wait_for_everyone()

    # Final save consistent with SFT saving approach
    if accelerator.is_main_process:
        final_unwrap = accelerator.unwrap_model(policy_value_model)
        accelerator.print(f"Saving final PPO model to {args.output_dir}")
        final_unwrap.base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(final_unwrap.state_dict(), os.path.join(args.output_dir, "ppo_policy_value_model.pt"))
        accelerator.print("All done! Check output in:", args.output_dir)


if __name__ == "__main__":
    main()