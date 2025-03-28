#!/usr/bin/env python3
"""
ReFT PPO script for a LLaMA-based model (text-only),
using Option 2: Keep a separate reference to the base LM unwrapped,
so we can call raw_lm.generate(...) even after DDP wrapping policy_value_model.
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8,9"
import json
import math
import random
import re
import copy

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil

from typing import Optional
from collections import deque
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from prompt_manager import PromptManager
from trl import AutoModelForCausalLMWithValueHead

def get_kl_penalty(
    policy_value_model,
    ref_base,
    input_ids_padded,
    attn_mask_padded,
    batch_lens,
    train_mask,
    kl_coef,
    cot_kl_discount
):
    """
    Calculate KL divergence penalty between policy and reference model.
    
    Args:
        policy_value_model: The current policy model
        ref_base: Reference model used as KL target
        input_ids_padded: Padded token IDs tensor [batch_size, seq_len]
        attn_mask_padded: Attention mask tensor [batch_size, seq_len]
        batch_lens: List of actual sequence lengths
        train_mask: Mask indicating which tokens to train on [batch_size, seq_len]
        kl_coef: KL penalty coefficient
        cot_kl_discount: Discount factor for KL on chain-of-thought tokens
        
    Returns:
        kl_penalty: Tensor of per-token KL penalties [batch_size, seq_len]
    """
    batch_size = input_ids_padded.size(0)
    kl_penalty = torch.zeros_like(input_ids_padded, dtype=torch.float, device=input_ids_padded.device)
    
    if ref_base is None or kl_coef <= 0:
        return kl_penalty
        
    with torch.no_grad():
        # Get policy logits
        policy_out = policy_value_model(
            input_ids=input_ids_padded,
            attention_mask=attn_mask_padded,
            use_cache=False
        )
        policy_logits = policy_out.logits if hasattr(policy_out, 'logits') else policy_out[0]
        
        # Get reference logits
        ref_out = ref_base(
            input_ids=input_ids_padded,
            attention_mask=attn_mask_padded,
            use_cache=False
        )
        ref_logits = ref_out.logits if hasattr(ref_out, 'logits') else ref_out[0]
        
        # Calculate log probabilities
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Calculate KL for each token
        for b in range(batch_size):
            seq_len_b = batch_lens[b]
            if seq_len_b <= 1:  # Skip if sequence is too short
                continue
            
            for t in range(seq_len_b - 1):
                token_next = input_ids_padded[b, t+1]
                policy_lp = policy_log_probs[b, t, token_next]
                ref_lp = ref_log_probs[b, t, token_next]
                token_kl = policy_lp.exp() * (policy_lp - ref_lp)
                
                # Apply KL coefficient, possibly discounted for CoT tokens
                kl_multiplier = kl_coef
                if train_mask[b, t] > 0 and t < seq_len_b - 2:
                    kl_multiplier = kl_coef * cot_kl_discount
                
                kl_penalty[b, t] = token_kl * kl_multiplier
                
    return kl_penalty

###############################################################################
# 1) PPO REWARD + ADVANTAGE UTILS
###############################################################################
def compute_logprobs_from_logits(logits, labels):
    """
    Compute log probabilities for each token from logits.
    
    Args:
        logits: Model output logits of shape [batch_size, seq_len, vocab_size]
        labels: Token ID labels of shape [batch_size, seq_len]
        
    Returns:
        token_logprobs: Log probabilities for each token [batch_size, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    labels_flat = labels.unsqueeze(-1)
    token_logprobs = torch.gather(log_probs, dim=-1, index=labels_flat).squeeze(-1)
    return token_logprobs


def gaussian_reward(
        pred_value: float,
        true_value: float,
        sigma: float = 5.0,
        near_exact_range: float = 5.0,
        near_exact_bonus: float = 20.0
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


def scaled_reward(pred_value: float, true_value: float, threshold=10.0, close_bonus_threshold=4.0, close_bonus=1.0, accelerator=None):
    """
    A simple numeric reward function for how close 'pred_value' is to 'true_value',
    giving up to 1.0 if within 'threshold' range, plus an additional bonus if very close.
    
    Args:
        pred_value: The predicted carbohydrate value
        true_value: The true carbohydrate value
        threshold: Maximum difference for non-zero reward (linear scaling)
        close_bonus_threshold: Threshold for additional bonus (e.g., 4 grams)
        close_bonus: Extra reward for predictions within close_bonus_threshold
        accelerator: Optional accelerator for distributed logging
        
    Returns:
        Reward value between 0.0 and (1.0 + close_bonus)
    """
    diff = abs(pred_value - true_value)
    
    # Start with base reward (linear scaling from threshold to 0)
    if diff >= threshold:
        base_reward = 0.0
    else:
        # Linear scale from 1.0 down to 0.0 as diff goes 0..threshold
        base_reward = 2.0 - (diff / threshold)
    
    # Add bonus for very accurate predictions
    bonus = close_bonus if diff <= close_bonus_threshold else 0.0
    
    return base_reward + bonus


def compute_gae_advantages(rewards, values, gamma=0.99, lam=0.95, mask=None):
    """
    Compute Generalized Advantage Estimation across token sequences.
    
    Args:
        rewards: Tensor of shape [batch_size, seq_len] with rewards
        values: Tensor of shape [batch_size, seq_len] with value predictions
        gamma: Discount factor
        lam: GAE lambda parameter
        mask: Optional attention mask to identify valid tokens
    
    Returns:
        advantages: Tensor of shape [batch_size, seq_len]
        returns: Tensor of shape [batch_size, seq_len]
    """
    batch_size = rewards.size(0)
    seq_len = rewards.size(1)
    
    # Create a proper mask if none is provided
    if mask is None:
        mask = torch.ones_like(rewards)
    else:
        # Ensure mask has the same shape as rewards and values
        if mask.size(1) != seq_len:
            # Create a new mask with the same shape as rewards
            new_mask = torch.zeros_like(rewards)
            # Copy over the values from the original mask, up to its length
            for b in range(batch_size):
                mask_len = min(mask.size(1), seq_len)
                new_mask[b, :mask_len] = mask[b, :mask_len]
            mask = new_mask
    
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # Process each batch item separately
    for b in range(batch_size):
        # Process tokens in reverse order for GAE calculation
        gae = 0
        for t in reversed(range(seq_len)):
            # Skip if this token is masked out
            if mask[b, t] == 0:
                continue
                
            # For last token or if next token is masked, bootstrap is 0
            if t == seq_len - 1 or mask[b, t+1] == 0:
                next_value = 0
            else:
                next_value = values[b, t+1]
            
            # GAE calculation formula
            delta = rewards[b, t] + gamma * next_value - values[b, t]
            gae = delta + gamma * lam * gae
            
            # Store results
            advantages[b, t] = gae
            returns[b, t] = gae + values[b, t]
    
    return advantages, returns

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
    policy_value_model,
    tokenizer,
    prompts,
    true_values,
    prompt_manager,
    max_new_tokens=200,
    accelerator=None,
    temperature=1.0,
    do_sample=False,
    kl_coef=0.02,
    cot_kl_discount=0.5,
    step_reward=0.000001,
    ref_base=None
):
    """
    Generate text and create properly padded tensors with consistent dimensions.
    Includes KL penalty calculation and properly handles batch dimensions.
    """
    device = next(policy_value_model.parameters()).device
    batch_size = len(prompts)

    # 1) Build prompts
    training_prompts = [prompt_manager.build_prompt(p, mode="inference") for p in prompts]
    enc = tokenizer(training_prompts, return_tensors="pt", padding=True, truncation=True)
    base_input_ids = enc["input_ids"].to(device)
    base_attention_mask = enc["attention_mask"].to(device)

    # 2) Generate text with unwrapped policy_value_model
    with torch.no_grad():
        # Unwrap the model before calling generate
        unwrapped_policy = accelerator.unwrap_model(policy_value_model)
        
        if do_sample:
            new_ids = unwrapped_policy.generate(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            new_ids = unwrapped_policy.generate(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

    # 3) Convert to text
    final_texts = [
        tokenizer.decode(seq, skip_special_tokens=False) for seq in new_ids
    ]

    # After generating and getting final_texts, add logging
    # Log a sample of generations (limit to 2 examples to avoid flooding logs)
    if accelerator and accelerator.is_main_process:
        num_examples = min(2, len(final_texts))
        accelerator.print("\n" + "="*80)
        accelerator.print(f"SAMPLE GENERATIONS (temp={temperature}, do_sample={do_sample}):")
        for i in range(num_examples):
            accelerator.print(f"\nEXAMPLE {i+1}:")
            accelerator.print(f"PROMPT: {prompts[i][:100]}...")
            accelerator.print(f"RESPONSE:\n{final_texts[i]}")
        accelerator.print("="*80 + "\n")

    # 4) Parse each response, compute numeric reward
    # Initialize lists with proper batch size
    parsing_success = [False] * batch_size  # Pre-allocate with correct size
    cots = [None] * batch_size              # Pre-allocate with correct size
    pred_values = [None] * batch_size       # Pre-allocate with correct size
    reward_list = [0.0] * batch_size        # Pre-allocate with correct size
    
    # Process each item in the batch
    for i in range(batch_size):
        text = final_texts[i]
        # Parse chain-of-thought from the text
        cot, ans_val = prompt_manager.parse_cot_and_answer(text)
        cots[i] = cot  # Use index assignment instead of append
        
        # Convert numeric value to float and calculate reward
        is_malformed = False
        try:
            guess = float(ans_val)
            parsing_success[i] = True  # Already using index assignment
        except ValueError:
            guess = None
            parsing_success[i] = False  # Already using index assignment
            is_malformed = True
            
        pred_values[i] = guess  # Use index assignment instead of append
        
        if guess is None or is_malformed:
            numeric_reward = -5.0
        else:
            true_val = float(true_values[i])
            numeric_reward = gaussian_reward(guess, true_val, sigma=5.0, near_exact_range=5.0, near_exact_bonus=40.0)
            
        reward_list[i] = numeric_reward  # Use index assignment instead of append
        
        # Add detailed logging for comparison
        if accelerator and accelerator.is_main_process and i < 3:  # Log first few examples
            accelerator.print(f"  Final prediction: {guess}")
            accelerator.print(f"  True value: {true_values[i]}")
            accelerator.print(f"  Reward: {numeric_reward}")
            accelerator.print(f"  Is malformed: {is_malformed}")

    # 5) Build padded tensors with consistent dimensions
    batch_lens = [seq.shape[0] for seq in new_ids]
    max_seq_len = max(batch_lens)

    input_ids_padded = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    attn_mask_padded = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)
    train_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)
    reward_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)

    # Fill padded tensors
    for b in range(batch_size):
        seq_len_b = batch_lens[b]
        if seq_len_b > 0:  # Guard against zero-length sequences
            input_ids_padded[b, :seq_len_b] = new_ids[b]
            attn_mask_padded[b, :seq_len_b] = 1.0

            # Mark tokens for training (not including the prompt)
            prompt_tokens = tokenizer(training_prompts[b], return_tensors="pt").input_ids[0]
            prompt_len = min(len(prompt_tokens), seq_len_b)
            if prompt_len < seq_len_b:
                train_mask[b, prompt_len:seq_len_b] = 1.0
                
            # Only apply step reward if prediction was successful
            if parsing_success[b] and pred_values[b] is not None:
                reward_tensor[b, :seq_len_b] = step_reward
            else:
                # Zero step reward for malformed outputs
                reward_tensor[b, :seq_len_b] = 0.0
            
            # Apply the main reward to the last token ONLY if sequence has length > 0
            reward_tensor[b, seq_len_b-1] += reward_list[b]

    # 6) Calculate KL penalty if reference model is provided
    kl_penalty = get_kl_penalty(
        policy_value_model=policy_value_model,
        ref_base=ref_base,
        input_ids_padded=input_ids_padded,
        attn_mask_padded=attn_mask_padded,
        batch_lens=batch_lens,
        train_mask=train_mask,
        kl_coef=kl_coef,
        cot_kl_discount=cot_kl_discount
    )
    
    # Apply KL penalty to reward
    reward_tensor = reward_tensor - kl_penalty

    # Return all required tensors and metadata
    return (
        input_ids_padded,    # [B, max_seq_len]
        attn_mask_padded,    # [B, max_seq_len]
        train_mask,          # [B, max_seq_len]
        reward_tensor,       # [B, max_seq_len]
        final_texts,
        cots,
        pred_values,
        training_prompts,
        parsing_success,
        kl_penalty
    )

###############################################################################
# 4) PPO STEP
###############################################################################
def ppo_step(
    model,
    policy_optimizer,
    value_optimizer, 
    old_logprobs,
    old_values,
    input_ids,
    attn_mask,
    train_mask,
    advantages,
    returns,
    clip_range=0.2,
    vf_coef=0.6
):
    """
    Run a PPO update with proper policy ratio handling.
    """
    # Forward pass
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    
    # Handle different output formats
    if isinstance(out, tuple) and len(out) >= 3:
        lm_logits, _, new_values = out
    elif hasattr(out, 'logits') and hasattr(out, 'hidden_states'):
        lm_logits, new_values = out.logits, out.hidden_states
    else:
        raise ValueError(f"Unexpected output format from model: {type(out)}")
    
    # Make sure new_values has correct shape
    if new_values.dim() == 3 and new_values.size(2) == 1:
        new_values = new_values.squeeze(-1)

    # Compute log probs for current policy
    new_logprobs = compute_logprobs_from_logits(
        lm_logits,  # no slicing
        input_ids   # match the same shape
    )

    # Combine attention mask with train mask
    valid_mask = attn_mask.float() * train_mask

    # Check for empty mask
    if valid_mask.sum() < 1.0:
        # No tokens to train on in this batch
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "mean_ratio": 1.0,
            "std_ratio": 0.0,
            "mean_advantage": 0.0,
            "std_advantage": 0.0
        }
    
    # IMPORTANT FIX: Allow ratio to exceed 1.0 by using a more appropriate clamp range
    delta_logprobs = new_logprobs - old_logprobs
    # Clamp for numerical stability but allow positive values
    delta_logprobs = torch.clamp(delta_logprobs, min=-5.0, max=5.0)
    ratio = torch.exp(delta_logprobs)  # Changed from delta_logprobs.exp()
    
    # Compute policy loss using clipped objective
    adv_masked = advantages * valid_mask
    ratio_masked = ratio * valid_mask

    # Compute policy loss
    pg_loss1 = -adv_masked * ratio_masked
    pg_loss2 = -adv_masked * torch.clamp(ratio_masked, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss_ = torch.max(pg_loss1, pg_loss2)
    
    mask_sum = valid_mask.sum().clamp_min(1.0)
    policy_loss = pg_loss_.sum() / mask_sum

    # Value loss with clipping
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
    vf_loss1 = (new_values - returns) ** 2
    vf_loss2 = (v_clipped - returns) ** 2
    vf_loss_ = 0.5 * torch.max(vf_loss1, vf_loss2)
    value_loss = (vf_loss_ * valid_mask).sum() / mask_sum

    # Calculate total loss for optimization
    total_loss = policy_loss + vf_coef * value_loss
    
    # FIXED: Single backward pass with combined loss
    # Zero gradients for both optimizers
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    
    # Single backward pass on the combined loss
    total_loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Step both optimizers
    policy_optimizer.step()
    value_optimizer.step()

    # Compute stats for logging
    with torch.no_grad():
        valid_mask_bool = (valid_mask > 0).bool()
        if valid_mask_bool.any():
            ratio_vals = ratio_masked[valid_mask_bool]
            adv_vals = adv_masked[valid_mask_bool]
            mean_ratio = ratio_vals.mean()
            std_ratio = ratio_vals.std().clamp_min(1e-8)
            mean_adv = adv_vals.mean()
            std_adv = adv_vals.std().clamp_min(1e-8)
        else:
            mean_ratio = torch.tensor(1.0, device=input_ids.device)
            std_ratio = torch.tensor(0.0, device=input_ids.device)
            mean_adv = torch.tensor(0.0, device=input_ids.device)
            std_adv = torch.tensor(0.0, device=input_ids.device)

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "total_loss": total_loss.item(),
        "mean_ratio": mean_ratio.item(),
        "std_ratio": std_ratio.item(),
        "mean_advantage": mean_adv.item(),
        "std_advantage": std_adv.item()
    }


###############################################################################
# 5) MAIN
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm_start_model", type=str, default="/data/nmysore/models/fine_tuned_model",
                        help="Path to your *fine-tuned* model checkpoint.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate for the policy network (default: 5e-6)")
    parser.add_argument("--value_lr", type=float, default=1e-6,
                        help="Learning rate for the value head (default: 1e-6)")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=random.randint(1, 10000),
                        help="Random seed for reproducibility (default: random)")
    parser.add_argument("--data_fraction", type=float, default=0.08)
    parser.add_argument("--wandb_project", type=str, default="ppo_nutri_g3")
    parser.add_argument("--wandb_entity", type=str, default="nmysore-uc-santa-barbara")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do_sample", action="store_true", default="True", help="Use sampling instead of greedy.")
    parser.add_argument("--ppo_epochs", type=int, default=3,
                        help="Number of PPO update epochs per batch")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range parameter")
    parser.add_argument("--cot_kl_discount", type=float, default=0.5,
                        help="Discount factor for KL penalty on CoT tokens (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, default="/data/nmysore/checkpoints/ppo_trained_model",
                        help="Where to store the final PPO model (so inference can load it).")
    parser.add_argument("--vf_coef", type=float, default=0.1,
                        help="Value function coefficient for PPO")
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

    # Create output directory if it doesn't exist
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save a copy of the args for future reference
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # After wandb.init add checkpoint tracking variables
    best_reward = -float('inf')
    last_save_step = 0
    save_interval = 100  # Save every 100 steps
    saved_checkpoints = []  # List to track saved checkpoints
    max_checkpoints_to_keep = 2  # Keep only 2 latest checkpoints

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

    config = AutoConfig.from_pretrained(args.warm_start_model)
    config.use_cache = False

    # reference model
    accelerator.print("Loading reference model from the same checkpoint")
    ref_base = LlamaForCausalLM.from_pretrained(
        args.warm_start_model,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    ref_base = ref_base.to(accelerator.device)  # Move ref_base to the same device
    for p in ref_base.parameters():
        p.requires_grad = False

    # 3) Build policy-value model using TRL
    accelerator.print("Loading policy_value_model as TRL AutoModelForCausalLMWithValueHead")
    policy_value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.warm_start_model,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )

    # ----------------------------------------------------------------------
    # FREEZE ALL LAYERS EXCEPT THE LAST 2
    # ----------------------------------------------------------------------
    # Count total layers first
    if hasattr(policy_value_model.pretrained_model, "model") and \
       hasattr(policy_value_model.pretrained_model.model, "layers"):
        total_layers = len(policy_value_model.pretrained_model.model.layers)
        accelerator.print(f"Model has {total_layers} transformer layers, freezing all but last 2")
    else:
        total_layers = 0
        accelerator.print("WARNING: Could not find transformer layers in model structure")

    # Track trainable parameters before freezing
    total_params_before = sum(p.numel() for p in policy_value_model.parameters() if p.requires_grad)

    # Freeze parameters by layer index
    for name, param in policy_value_model.pretrained_model.named_parameters():
        # Check if this param belongs to a particular transformer block
        match = re.search(r"model\.layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            # If it's from layers < total_layers - 6, freeze:
            if layer_idx < total_layers - 6:
                param.requires_grad = False
        else:
            # Also freeze embeddings & other modules outside the transformer blocks
            param.requires_grad = False

    # Keep the value head trainable - it's outside the pretrained_model
    for name, param in policy_value_model.named_parameters():
        if 'pretrained_model' not in name:  # Value head parameters
            param.requires_grad = True

    # Track trainable parameters after freezing
    total_params_after = sum(p.numel() for p in policy_value_model.parameters() if p.requires_grad)
    accelerator.print(f"Trainable parameters reduced from {total_params_before:,} to {total_params_after:,}")
    accelerator.print(f"Now training only {total_params_after/total_params_before:.2%} of parameters")
    # ----------------------------------------------------------------------

    # 4) Load tokenizer
    accelerator.print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.warm_start_model)
    
    # Always set padding_side to 'left' for decoder-only models
    tokenizer.padding_side = "left"
    
    # Set pad token if needed (but always ensure left padding)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Verify the special tokens exist (for debugging)
    accelerator.print("Verifying special tokens in tokenizer vocabulary:")
    expected_tokens = [
        "<|begin_cot|>", "<|end_cot|>"
    ]
    
    for token in expected_tokens:
        if token not in tokenizer.get_vocab():
            print(f"WARNING: Token '{token}' not found in vocabulary! Chain-of-thought parsing may fail.")
        else:
            accelerator.print(f"Token {token} found with ID: {tokenizer.convert_tokens_to_ids(token)}")
            
    # If tokens are missing, add them now (not ideal but prevents errors)
    missing_tokens = [t for t in expected_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if missing_tokens:
        accelerator.print(f"Adding {len(missing_tokens)} missing special tokens to the tokenizer")
        special_tokens = {"additional_special_tokens": missing_tokens}
        num_added = tokenizer.add_special_tokens(special_tokens)
        
        # Resize all models' embeddings
        policy_value_model.pretrained_model.resize_token_embeddings(len(tokenizer))
        ref_base.resize_token_embeddings(len(tokenizer))
        
        accelerator.print("Token embeddings have been resized. Note: newly added token embeddings are random!")

    # Wrap for DDP
    policy_value_model, ref_base = accelerator.prepare(policy_value_model, ref_base)

    # 5) optimizer
    # Split parameters into policy and value groups
    policy_params = []
    value_params = []
    
    for name, param in policy_value_model.named_parameters():
        if "v_head" in name or "value_head" in name:
            value_params.append(param)
        else:
            policy_params.append(param)
    
    # Create separate optimizers
    policy_optimizer = torch.optim.AdamW(policy_params, lr=args.lr)
    value_optimizer = torch.optim.AdamW(value_params, lr=args.value_lr)
    
    # Separate schedulers for policy and value optimizers
    policy_sched = get_constant_schedule_with_warmup(policy_optimizer, num_warmup_steps=0)
    # Linear schedule with warmup for value head
    value_sched = get_linear_schedule_with_warmup(
        value_optimizer, 
        num_warmup_steps=100,  # Adjust this as needed
        num_training_steps=len(dataset) // args.batch_size * args.n_epochs
    )

    global_step = 0
    reward_deque = deque(maxlen=50)

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

            # Create a frozen copy of the current policy to serve as old policy
            old_policy = copy.deepcopy(policy_value_model).eval()

            # Rollout with OLD policy
            (
                input_ids_padded,    # [B, max_seq_len]
                attn_mask_padded,    # [B, max_seq_len] 
                train_mask,          # [B, max_seq_len]
                reward_tensor,       # [B, max_seq_len]
                final_texts,
                cots,
                pred_values,
                training_prompts,    # We need this to re-tokenize if necessary
                parsing_success,
                kl_penalty
            ) = rollout_step(
                old_policy,          # Use OLD policy for generation
                tokenizer,
                b_q,
                b_a,
                prompt_manager,
                max_new_tokens=args.max_new_tokens,
                accelerator=accelerator,  # Make sure to pass accelerator
                temperature=args.temperature,
                do_sample=args.do_sample,
                kl_coef=args.kl_coef,
                cot_kl_discount=args.cot_kl_discount,
                step_reward=0.000001,
                ref_base=ref_base
            )

            # Get average reward from last tokens for monitoring
            batch_sz = len(b_q)
            last_rewards = []
            avg_kl = 0.0
            kl_count = 0
            for b in range(batch_sz):
                seq_len = int(attn_mask_padded[b].sum().item())
                if seq_len > 0:
                    last_rewards.append(reward_tensor[b, seq_len-1].item())
                    
                    # KL penalty tracking if available
                    if kl_penalty is not None and kl_penalty.shape[0] > b:
                        kl_sum = kl_penalty[b].sum().item()
                        if kl_sum > 0:
                            avg_kl += kl_sum
                            kl_count += 1
            
            # Calculate statistics
            avg_reward = np.mean(last_rewards) if last_rewards else 0.0
            reward_deque.append(avg_reward)
            avg_kl = avg_kl / max(1, kl_count)
            avg_pool_reward = np.mean(reward_deque)
            
            # Log detailed stats of generation results
            if accelerator.is_main_process:
                # Print summary of batch results
                success_rate = sum(parsing_success) / len(parsing_success) * 100
                accelerator.print(f"\nBatch Summary - Step {global_step}:")
                accelerator.print(f"  Parsing success rate: {success_rate:.1f}%")
                accelerator.print(f"  Avg reward: {avg_reward:.4f} (rolling avg: {avg_pool_reward:.4f})")
                
                # Log some examples to wandb if enabled
                if wandb.run:
                    # Log a few examples with their parsed values
                    log_samples = min(3, len(final_texts))
                    for i in range(log_samples):
                        # Get reward from the last token of reward_tensor for this example
                        seq_len = int(attn_mask_padded[i].sum().item())
                        last_token_reward = reward_tensor[i, seq_len-1].item() if seq_len > 0 else 0.0
                        
                        example = {
                            "prompt": b_q[i],
                            "response": final_texts[i],
                            "parsed_value": pred_values[i],
                            "true_value": b_a[i],
                            "reward": last_token_reward,  # Use the reward from reward_tensor
                            "parsing_success": parsing_success[i]
                        }
                        wandb.log({f"examples/example_{i}": example}, step=global_step)

            # Get log probabilities and values from the old policy
            old_logprobs, old_values = get_log_probs_and_values(
                policy=old_policy,
                input_ids=input_ids_padded,
                attention_mask=attn_mask_padded,
                use_cache=False
            )

            # [B] Compute advantages & returns from the old values
            advantages, returns = compute_gae_advantages(
                rewards=reward_tensor,
                values=old_values,
                mask=attn_mask_padded * train_mask,
                gamma=0.99,
                lam=0.95
            )

            # [B.1] Advantage normalization (optional but often helpful):
            with torch.no_grad():
                valid_mask = (attn_mask_padded * train_mask) > 0
                if valid_mask.any():
                    adv_valid = advantages[valid_mask]
                    mean_adv = adv_valid.mean()
                    std_adv = adv_valid.std().clamp_min(1e-8)
                    # Normalize in place
                    advantages.sub_(mean_adv).div_(std_adv)

            # [C] PPO epochs (multiple updates per batch)
            for ppo_epoch in range(args.ppo_epochs):
                loss_dict = ppo_step(
                    model=policy_value_model,     # Use current policy for updates
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer, 
                    old_logprobs=old_logprobs,    # just computed
                    old_values=old_values,        # just computed
                    input_ids=input_ids_padded,
                    attn_mask=attn_mask_padded,
                    train_mask=train_mask,
                    advantages=advantages,
                    returns=returns,
                    clip_range=args.clip_range,
                    vf_coef=args.vf_coef
                )
                
                # Step both schedulers
                policy_sched.step()
                value_sched.step()
                
            global_step += 1

            accelerator.wait_for_everyone()

            # Comprehensive logging at the end of this batch
            if accelerator.is_main_process and wandb.run and loss_dict is not None:
                wandb.log({
                    "train/epoch": ep,
                    "train/step": global_step,
                    "train/avg_reward": avg_reward,
                    "train/rolling_avg_reward": avg_pool_reward,
                    "train/policy_loss": loss_dict["policy_loss"],
                    "train/value_loss": loss_dict["value_loss"],
                    "train/total_loss": loss_dict["total_loss"],
                    "train/mean_ratio": loss_dict["mean_ratio"],
                    "train/std_ratio": loss_dict["std_ratio"],
                    "train/mean_advantage": loss_dict["mean_advantage"],
                    "train/std_advantage": loss_dict["std_advantage"],
                    "train/avg_kl": avg_kl
                }, step=global_step)
                
                # Print summary with KL info
                accelerator.print(f"  KL coefficient: {args.kl_coef:.6f}, Avg KL: {avg_kl:.6f}")
                if avg_kl > 0.05:  # Arbitrary threshold
                    accelerator.print("  ⚠️ WARNING: High KL divergence detected!")

            # Save model periodically and when we get best reward
            if accelerator.is_main_process:
                # Check if we should save based on interval
                save_by_interval = (global_step - last_save_step) >= save_interval
                
                # Check if we should save based on reward improvement
                save_by_reward = avg_pool_reward > best_reward
                
                if save_by_interval or save_by_reward:
                    # Create checkpoint directory
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Get unwrapped model
                    unwrapped_model = accelerator.unwrap_model(policy_value_model)
                    
                    # Save model
                    accelerator.print(f"Saving checkpoint to {checkpoint_dir}")
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Save optimizer state
                    torch.save(policy_optimizer.state_dict(), os.path.join(checkpoint_dir, "policy_optimizer.pt"))
                    torch.save(value_optimizer.state_dict(), os.path.join(checkpoint_dir, "value_optimizer.pt"))
                    
                    # Save training state
                    torch.save({
                        "global_step": global_step,
                        "rolling_avg_reward": avg_pool_reward,
                        "best_reward": best_reward,
                        "epoch": ep,
                    }, os.path.join(checkpoint_dir, "training_state.pt"))
                    
                    # Update tracking variables
                    last_save_step = global_step
                    saved_checkpoints.append(checkpoint_dir)  # Add to saved list
                    
                    if save_by_reward:
                        best_reward = avg_pool_reward
                        # Mark this as best checkpoint with a symlink
                        best_link = os.path.join(args.output_dir, "checkpoint-best")
                        if os.path.exists(best_link):
                            if os.path.islink(best_link):
                                os.unlink(best_link)
                            else:
                                shutil.rmtree(best_link)
                        # Create relative symlink
                        os.symlink(os.path.basename(checkpoint_dir), best_link)
                        accelerator.print(f"New best model with reward {best_reward:.4f}")
                    
                    # Cleanup old checkpoints, keeping only the most recent ones
                    if len(saved_checkpoints) > max_checkpoints_to_keep:
                        # Get the real path of the best checkpoint
                        best_ckpt_path = None
                        if os.path.exists(os.path.join(args.output_dir, "checkpoint-best")):
                            best_ckpt_path = os.path.realpath(os.path.join(args.output_dir, "checkpoint-best"))
                        
                        # Sort checkpoints by creation time (oldest first)
                        to_remove = saved_checkpoints[:-max_checkpoints_to_keep]
                        
                        for old_ckpt in to_remove:
                            # Don't delete if it's the best checkpoint
                            if best_ckpt_path and os.path.realpath(old_ckpt) == best_ckpt_path:
                                accelerator.print(f"Keeping best checkpoint: {old_ckpt}")
                                continue
                            
                            # Delete the old checkpoint
                            accelerator.print(f"Removing old checkpoint: {old_ckpt}")
                            if os.path.exists(old_ckpt):
                                try:
                                    shutil.rmtree(old_ckpt)
                                except Exception as e:
                                    accelerator.print(f"Error removing checkpoint {old_ckpt}: {e}")
                        
                        # Update our saved_checkpoints list, keeping only the ones we didn't delete
                        saved_checkpoints = saved_checkpoints[-max_checkpoints_to_keep:]

        accelerator.print(f"Epoch={ep} completed. Rolling avg reward={avg_pool_reward:.3f}")

    accelerator.print("Done PPO training!")
    accelerator.wait_for_everyone()

    # Final model save
    if accelerator.is_main_process:
        accelerator.print(f"Training complete! Saving final model to {args.output_dir}")
        final_model = accelerator.unwrap_model(policy_value_model)
        final_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Save final training state
        torch.save({
            "global_step": global_step,
            "rolling_avg_reward": avg_pool_reward,
            "best_reward": best_reward,
            "total_epochs": args.n_epochs,
        }, os.path.join(args.output_dir, "final_training_state.pt"))
        
        accelerator.print(f"Final model saved. Best reward achieved: {best_reward:.4f}")

def get_log_probs_and_values(policy, input_ids, attention_mask, use_cache=False):
    """
    Perform a forward pass with a policy model and extract log probabilities and values.
    
    Args:
        policy: The policy model to evaluate
        input_ids: Tensor of token IDs [batch_size, seq_len]
        attention_mask: Attention mask for padding [batch_size, seq_len]
        use_cache: Whether to use KV cache for faster inference
        
    Returns:
        tuple: A tuple containing:
            - logprobs: Log probabilities for each token [batch_size, seq_len]
            - values: Value estimates [batch_size, seq_len]
    """
    with torch.no_grad():
        # 1) Forward pass on the policy
        out = policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        
        # 2) Extract logits + values
        if isinstance(out, tuple) and len(out) >= 3:
            lm_logits, _, values = out
        elif hasattr(out, 'logits') and hasattr(out, 'hidden_states'):
            lm_logits, values = out.logits, out.hidden_states
        else:
            raise ValueError(f"Unexpected output format from policy: {type(out)}")

        # Ensure correct shape
        if values.dim() == 3 and values.size(2) == 1:
            values = values.squeeze(-1)

        # 3) Compute logprobs from lm_logits
        logprobs = compute_logprobs_from_logits(
            lm_logits,  # no slicing
            input_ids   # match the same shape
        )
        
    return logprobs, values

if __name__ == "__main__":
    main()