#!/usr/bin/env python3
"""
ReFT PPO script for Gemma 3 1B (text-only),
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
    Gemma3ForCausalLM,
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


def scaled_reward(pred_value: float, true_value: float, threshold=5.0, accelerator=None):
    """
    A simple numeric reward function for how close 'pred_value' is to 'true_value',
    giving up to 1.0 if within 'threshold', else 0.0
    """
    if accelerator:
        accelerator.print(f"Pred={pred_value}, True={true_value}")
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
    Wraps Gemma3ForCausalLM + an extra value head in text-only mode.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        # minimal value head
        self.v_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
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
# 3) ROLLOUT STEP (Multi-token generation) using "raw_lm"   (★)
###############################################################################
def rollout_step(
        policy_value_model: PolicyValueModel,
        raw_lm,                          # (★) pass the unwrapped LM for .generate()
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

    # Build inference prompts
    inference_prompts = [prompt_manager.build_prompt(p) for p in prompts]
    enc = tokenizer(inference_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # --- 1) Generate final sequences with raw_lm ---
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

    # decode preserving special tokens
    final_texts = [
        tokenizer.decode(seq, skip_special_tokens=False)
        for seq in outputs
    ]

    # --- 2) Parse for <cot> and <answer>, compute reward ---
    reward_list = []
    for i, text in enumerate(final_texts):
        has_cot = ("<cot>" in text) and ("</cot>" in text)
        has_answer = ("<answer>" in text) and ("</answer>" in text)
        has_output_dict = False
        if has_answer and "Output: {\"total_carbohydrates\":" in text:
            has_output_dict = True
        correct_structure = has_cot and has_answer and has_output_dict
        structure_reward = 0.2 if correct_structure else 0.0

        chain_of_thought, final_ans = prompt_manager.parse_cot_and_answer(
            text,
            verbose=accelerator is not None,
            log_fn=accelerator.print if accelerator else None
        )
        guess = prompt_manager.extract_carbs_from_answer(final_ans)
        if guess is not None:
            numeric_reward = scaled_reward(guess, true_values[i], threshold=20.0, accelerator=accelerator)
        else:
            numeric_reward = -1.0

        reward_val = numeric_reward + structure_reward
        reward_list.append(reward_val)

    # --- 3) Re-encode the entire text to compute logprobs + values from the policy_value_model (DDP) ---
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
    for b in range(batch_size):
        last_idx = seq_lens[b].item() - 1
        reward_tensor[b, last_idx] = reward_list[b]

    # train_mask => newly generated portion
    prompt_lens = attention_mask.sum(dim=1)
    train_mask = torch.zeros_like(full_input_ids, dtype=torch.float)
    for b in range(batch_size):
        start = int(prompt_lens[b].item())
        end = int(seq_lens[b].item())
        train_mask[b, start:end] = 1.0

    return (
        full_input_ids,
        full_attention_mask,
        old_logprobs,
        values_full,
        reward_tensor,
        train_mask,
        reward_list
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
        kl_coef=0.02,
        clip_range=0.2,
        vf_coef=1.0
):
    """
    Run a PPO update on a batch of trajectories.
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
    mask_sum = valid_mask.sum()
    policy_loss = pg_loss_.sum() / mask_sum.clamp_min(1.0)

    # Value Loss
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
    vf_loss1 = (new_values - returns) ** 2
    vf_loss2 = (v_clipped - returns) ** 2
    vf_loss_ = 0.5 * torch.max(vf_loss1, vf_loss2)
    value_loss = (vf_loss_ * valid_mask).sum() / mask_sum.clamp_min(1.0)

    # KL vs reference model
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
    kl_per_token = kl_per_token * valid_mask
    kl_mean = kl_per_token.sum() / mask_sum.clamp_min(1.0)
    kl_loss = kl_coef * kl_mean

    total_loss = policy_loss + vf_coef * value_loss + kl_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_value": kl_mean.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item()
    }


###############################################################################
# 5) EVAL  (★) Also call raw_lm.generate instead of DDP, for consistency
###############################################################################
def evaluate_on_test_set(
        policy_value_model,
        raw_lm,  # unwrapped LM
        tokenizer,
        test_data,
        prompt_manager,
        batch_size=4,
        accelerator=None
):
    """
    Evaluate by generating from raw_lm, then parse the final text for carbs.
    """
    device = next(policy_value_model.parameters()).device
    policy_value_model.eval()
    raw_lm.eval()

    mae_list = []
    idx = 0
    while idx < len(test_data):
        subset = test_data[idx: idx + batch_size]
        idx += batch_size
        questions = [item["question"] for item in subset]
        true_vals = [float(item["answer_value"]) for item in subset]

        # Use inference prompt for consistency
        inference_prompts = [prompt_manager.build_inference_prompt(q) for q in questions]
        enc = tokenizer(inference_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            # multi-token generation using raw_lm
            outputs = raw_lm.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )

        gen_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in outputs]

        for i, text in enumerate(gen_texts):
            chain_of_thought, final_ans = prompt_manager.parse_cot_and_answer(
                text,
                verbose=False
            )
            guess = prompt_manager.extract_carbs_from_answer(final_ans)
            if guess is not None:
                mae_list.append(abs(guess - true_vals[i]))

    policy_value_model.train()
    raw_lm.train()
    if len(mae_list) == 0:
        return None
    return sum(mae_list) / len(mae_list)


###############################################################################
# 6) MAIN
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm_start_model", type=str, default="fine_tuned_model",
                        help="Path to your *fine-tuned* model checkpoint.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_fraction", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="ppo_nutri_g3")
    parser.add_argument("--wandb_entity", type=str, default="nmysore-uc-santa-barbara")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy.")
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

    test_data = None
    if args.test_file:
        with open(args.test_file, "r") as f:
            test_data = json.load(f)
        accelerator.print(f"Loaded test data, size={len(test_data)}")

    accelerator.print(f"Loaded {len(dataset)} training samples from {args.train_file}")

    # 2) (★) Load your fine-tuned model as raw_lm. Keep it unwrapped for .generate() calls
    accelerator.print(f"Loading raw_lm from {args.warm_start_model}")
    config = AutoConfig.from_pretrained(args.warm_start_model, trust_remote_code=True)
    if hasattr(config, "vision_config"):
        del config.vision_config
    config.use_cache = False

    raw_lm = Gemma3ForCausalLM.from_pretrained(
        args.warm_start_model,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # reference model
    accelerator.print("Loading reference model from same checkpoint")
    ref_base = Gemma3ForCausalLM.from_pretrained(
        args.warm_start_model,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    for p in ref_base.parameters():
        p.requires_grad = False

    # 3) Build policy-value model from raw_lm, but do not alter raw_lm for .generate
    policy_value_model = PolicyValueModel(raw_lm)
    policy_value_model.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

    # 4) Load tokenizer
    accelerator.print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.warm_start_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # (★) Wrap policy_value_model + ref_base in DDP, but NOT raw_lm
    policy_value_model, ref_base = accelerator.prepare(policy_value_model, ref_base)

    # 5) optimizer
    optimizer = torch.optim.AdamW(policy_value_model.parameters(), lr=args.lr)
    sched = get_constant_schedule_with_warmup(optimizer, 0)

    global_step = 0
    reward_deque = deque(maxlen=100)

    # Initialize PromptManager once
    prompt_manager = PromptManager()

    # 6) PPO
    ppo_epochs = 4
    for ep in range(args.n_epochs):
        random.shuffle(dataset)
        steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
        for step_i in range(steps_per_epoch):
            batch_slice = dataset[step_i * args.batch_size:(step_i + 1) * args.batch_size]
            if not batch_slice:
                break
            b_q = [x[0] for x in batch_slice]
            b_a = [x[1] for x in batch_slice]

            # multi-step rollout with raw_lm for generation
            (new_input_ids,
             new_attn_mask,
             old_logprobs,
             old_values,
             reward_tensor,
             train_mask,
             reward_list) = rollout_step(
                policy_value_model,
                raw_lm,                 # (★) pass the unwrapped model
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
            # multiple PPO update epochs on the same data
            last_loss_dict = None
            for _ in range(ppo_epochs):
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
                    kl_coef=args.kl_coef,
                    clip_range=0.1,
                    vf_coef=1.0
                )
                sched.step()
                last_loss_dict = loss_dict
            global_step += 1

            # logging
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
                    "train/total_loss": last_loss_dict["total_loss"]
                }, step=global_step)

            if test_data and accelerator.is_main_process and (global_step % 50 == 0):
                quick_mae = evaluate_on_test_set(
                    policy_value_model,
                    raw_lm,            # pass raw_lm for generation in eval
                    tokenizer,
                    test_data,
                    prompt_manager,
                    batch_size=args.batch_size,
                    accelerator=accelerator
                )
                if quick_mae is not None and wandb.run:
                    wandb.log({"test/quick_mae": quick_mae}, step=global_step)
                accelerator.print(f"[Step={global_step}] Rolling avg={rolling_avg:.3f}, quick mae={quick_mae}")

            if global_step % 200 == 0:
                accelerator.print(f"Saving checkpoint at step={global_step}")
                unwrap = accelerator.unwrap_model(policy_value_model)
                ckdir = f"ppo_out/checkpoint_{global_step}"
                unwrap.base_model.save_pretrained(ckdir)
                tokenizer.save_pretrained(ckdir)

        # end of epoch
        if test_data and accelerator.is_main_process:
            mae = evaluate_on_test_set(
                policy_value_model,
                raw_lm,
                tokenizer,
                test_data,
                prompt_manager,
                batch_size=args.batch_size,
                accelerator=accelerator
            )
            if mae and wandb.run:
                wandb.log({"test/mae": mae}, step=global_step)
            accelerator.print(f"Epoch={ep}, test mae={mae}")

    accelerator.print("Done PPO training!")
    accelerator.wait_for_everyone()
    final_unwrap = accelerator.unwrap_model(policy_value_model)
    final_unwrap.base_model.save_pretrained("ppo_trained_model")
    tokenizer.save_pretrained("ppo_trained_model")
    torch.save(final_unwrap.state_dict(), "ppo_policy_value_model.pt")


if __name__ == "__main__":
    main()