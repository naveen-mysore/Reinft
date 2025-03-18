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

def gaussian_reward(pred_value: float, true_value: float, sigma=10.0):
    """
    Reward in (0,1] using an exponential decay away from the ground truth.
    sigma controls how quickly the reward falls off.
    """
    diff = pred_value - true_value
    # Exponentially decaying reward; 1.0 if guess==true, ~0 if very far
    return math.exp(- (diff * diff) / (2.0 * sigma * sigma))

def scaled_reward(pred_value: float, true_value: float, threshold=5.0, accelerator=None):
    """
    A simple numeric reward function for how close 'pred_value' is to 'true_value',
    giving up to 1.0 if within 'threshold', else 0.0.
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
    Wraps Gemma3ForCausalLM + an extra value head in text-only mode.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        # Minimal value head
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
# 3) ROLLOUT STEP
###############################################################################
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
            do_sample=True,
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

    # 2) Parse <cot> and <answer>, compute reward
    reward_list = []
    for i, text in enumerate(final_texts):
        has_cot = ("<cot>" in text) and ("</cot>" in text)
        has_answer = ("<answer>" in text) and ("</answer>" in text)
        has_output_dict = False
        if has_answer and "Output: {\"total_carbohydrates\":" in text:
            has_output_dict = True
        correct_structure = has_cot and has_answer and has_output_dict
        structure_reward = 0.1 if correct_structure else 0.0

        chain_of_thought, final_ans = prompt_manager.parse_cot_and_answer(
            text,
            verbose=accelerator is not None,
            log_fn=accelerator.print if accelerator else None
        )
        guess = prompt_manager.extract_carbs_from_answer(final_ans)
        
        # Check if the answer is malformed - we need to be stricter
        # If there's a number but with garbage text, consider it malformed
        is_malformed = False
        if final_ans:
            # Check for proper JSON format (allowing only digits, decimal point, and quotes)
            clean_json = re.match(r'.*?"total_carbohydrates":\s*"?(\d+\.?\d*)"?\s*}.*?', final_ans)
            # Check for proper Output format (allowing only digits and decimal point)
            clean_output = re.match(r'Output:\s*(\d+\.?\d*)\s*$', final_ans)
            
            # Check if there's extra text after the number in the JSON format
            json_with_garbage = re.search(r'"total_carbohydrates":\s*"?(\d+\.?\d*)[^"}]*[^0-9\.\s"}]', final_ans)
            
            if (not (clean_json or clean_output)) or json_with_garbage:
                # The answer is not in a clean format or has garbage text
                is_malformed = True
        
        if guess is None or is_malformed:
            # No valid number found or answer is malformed
            numeric_reward = -0.05
        else:
            # Use the existing reward calculation logic
            numeric_reward = gaussian_reward(guess, true_values[i], sigma=30.0)
            #numeric_reward = scaled_reward(guess, true_values[i], threshold=50.0, accelerator=accelerator)

        reward_val = numeric_reward + structure_reward
        reward_list.append(reward_val)

    # 3) Re-encode to compute logprobs + values from policy_value_model
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

    # 4) Build CoT mask - a mask of 1.0 for tokens in <cot>...</cot>, 0.0 otherwise
    cot_mask = torch.zeros_like(full_input_ids, dtype=torch.float)
    
    # Helper function to find sublist in a list
    def find_sublist(sub, main):
        # returns start index of sub in main, or -1 if not found
        for i in range(len(main) - len(sub) + 1):
            if main[i:i+len(sub)] == sub:
                return i
        return -1
    
    for b in range(batch_size):
        txt = final_texts[b]
        # Find substring indices for the <cot> ... </cot> region
        start_cot = txt.find("<cot>")
        end_cot = txt.find("</cot>")
        if start_cot == -1 or end_cot == -1 or end_cot <= start_cot:
            # No CoT or malformed => skip
            continue

        # Get the CoT substring including tags
        cot_substring = txt[start_cot:end_cot + len("</cot>")]

        # Tokenize only that substring to get the exact tokens
        cot_enc = tokenizer(cot_substring, add_special_tokens=False)
        cot_token_ids = cot_enc["input_ids"]

        # Find the token sequence in the full sequence
        full_ids_list = full_input_ids[b].tolist()
        start_idx = find_sublist(cot_token_ids, full_ids_list)
        
        if start_idx != -1:
            end_idx = start_idx + len(cot_token_ids)
            # Set mask=1 for the CoT region
            cot_mask[b, start_idx:end_idx] = 1.0

    # Add this immediately after rollout_step is called
    if accelerator and accelerator.is_main_process:
        accelerator.print("\n----- RAW TEXT DEBUG -----")
        for i, text in enumerate(final_texts):
            accelerator.print(f"Example {i+1} Raw Text (truncated):")
            # Print only the first 200 chars to avoid overwhelming the console
            truncated = text[:200] + "..." if len(text) > 200 else text
            accelerator.print(truncated)
            accelerator.print("---")

    # Add this where we're processing the prediction results
    pred_values = []
    for i, text in enumerate(final_texts):
        if accelerator and accelerator.is_main_process:
            accelerator.print(f"\nDEBUG - Example {i+1} ----------------")
            accelerator.print(f"Raw text snippet: {text[:50]}...{text[-50:]}")
            
        # Attempt to parse with detailed debugging
        chain_of_thought, final_ans = prompt_manager.parse_cot_and_answer(
            text, 
            verbose=(accelerator and accelerator.is_main_process),
            log_fn=accelerator.print if accelerator else print
        )
        
        # Extract the carb value
        guess = prompt_manager.extract_carbs_from_answer(final_ans)
        
        if accelerator and accelerator.is_main_process:
            accelerator.print(f"CoT extracted: {len(chain_of_thought)} chars")
            accelerator.print(f"Answer extracted: '{final_ans}'")
            accelerator.print(f"Final numeric value: {guess}")
        
        pred_values.append(guess)

    return (
        full_input_ids,
        full_attention_mask,
        old_logprobs,
        values_full,
        reward_tensor,
        train_mask,
        reward_list,
        cot_mask,
        final_texts
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
    cot_kl_discount=0.5
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
    # Scale factor: 1.0 for normal tokens, cot_kl_discount for CoT tokens
    kl_scaling = (1.0 - cot_mask) + (cot_kl_discount * cot_mask)
    
    # Apply scaling to KL
    kl_per_token = kl_per_token * valid_mask * kl_scaling
    
    # Sum with respect to the scaled mask
    scaled_mask_sum = (valid_mask * kl_scaling).sum()
    kl_mean = kl_per_token.sum() / scaled_mask_sum.clamp_min(1.0)
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
# 5) MAIN
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm_start_model", type=str, default="fine_tuned_model",
                        help="Path to your *fine-tuned* model checkpoint.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--kl_coef", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--data_fraction", type=float, default=0.6)
    parser.add_argument("--wandb_project", type=str, default="ppo_nutri_g3")
    parser.add_argument("--wandb_entity", type=str, default="nmysore-uc-santa-barbara")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--do_sample", action="store_true", default="True", help="Use sampling instead of greedy.")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="Number of PPO update epochs per batch")
    parser.add_argument("--clip_range", type=float, default=0.6,
                        help="PPO clip range parameter")
    parser.add_argument("--cot_kl_discount", type=float, default=0.5,
                        help="Discount factor for KL penalty on CoT tokens (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, default="ppo_trained_model",
                        help="Where to store the final PPO model (so inference can load it).")
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

    # 3) Build policy-value model
    policy_value_model = PolicyValueModel(raw_lm)
    policy_value_model.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

    # 4) Load tokenizer
    accelerator.print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.warm_start_model, trust_remote_code=True)
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
                final_texts
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

            # Now parse the actual model outputs (not the queries):
            pred_values = []
            parsed_cot = []
            parsed_answer = []

            for i, gen_text in enumerate(final_texts):
                # Parse the generated text, not the query
                chain_of_thought, final_ans = prompt_manager.parse_cot_and_answer(gen_text)
                guess = prompt_manager.extract_carbs_from_answer(final_ans)
                
                pred_values.append(guess)
                parsed_cot.append(chain_of_thought)
                parsed_answer.append(final_ans)

            # Print detailed debug information
            if accelerator and accelerator.is_main_process:
                accelerator.print("\n----- Batch Predictions -----")
                for i in range(len(b_q)):
                    accelerator.print(f"Example {i+1}:")
                    # Print a truncated version of the query to save space
                    query_trunc = b_q[i][:50] + "..." if len(b_q[i]) > 50 else b_q[i]
                    accelerator.print(f"  Query: {query_trunc}")
                    
                    # Print truncated CoT if it's long
                    cot_trunc = parsed_cot[i][:100] + "..." if len(parsed_cot[i]) > 100 else parsed_cot[i]
                    accelerator.print(f"  Generated CoT: {cot_trunc}")
                    
                    # Print the answer block
                    accelerator.print(f"  Generated Answer: {parsed_answer[i]}")
                    
                    # Show the parsed value and ground truth
                    accelerator.print(f"  Pred={pred_values[i]}")
                    accelerator.print(f"  True={b_a[i]}")
                    accelerator.print(f"  Reward={reward_list[i]}")
                    accelerator.print("----------------------------")

            # Multiple PPO epochs
            last_loss_dict = None
            for _ in range(args.ppo_epochs):
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
                    cot_kl_discount=args.cot_kl_discount
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
                    "train/total_loss": last_loss_dict["total_loss"]
                }, step=global_step)

            # Optional checkpoint saving
            if global_step % 200 == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.print(f"Saving checkpoint at step={global_step}")
                    unwrap = accelerator.unwrap_model(policy_value_model)
                    ckdir = f"ppo_out/checkpoint_{global_step}"
                    # Save the base model and tokenizer
                    unwrap.base_model.save_pretrained(ckdir)
                    tokenizer.save_pretrained(ckdir)
                accelerator.wait_for_everyone()

        accelerator.print(f"Epoch={ep} completed. Rolling avg reward={rolling_avg:.3f}")

    accelerator.print("Done PPO training!")
    accelerator.wait_for_everyone()

    # Final save consistent with SFT saving approach
    if accelerator.is_main_process:
        final_unwrap = accelerator.unwrap_model(policy_value_model)
        # This ensures your inference script can do:
        # Gemma3ForCausalLM.from_pretrained(args.output_dir, trust_remote_code=True)
        accelerator.print(f"Saving final PPO model to {args.output_dir}")
        final_unwrap.base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # (Optional) Save the entire policy-value state dict for custom usage:
        torch.save(final_unwrap.state_dict(), os.path.join(args.output_dir, "ppo_policy_value_model.pt"))
        accelerator.print("All done! Check output in:", args.output_dir)

if __name__ == "__main__":
    main()