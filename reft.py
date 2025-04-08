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
import copy

import torch
import numpy as np
import torch.nn.functional as F
import shutil

from collections import deque
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from src.ppo.base_model import BaseModel
from src.ppo.policy_value import PolicyValueModelWrapper
from src.prompt_manager import PromptManager
from src.ppo.ppo_trainer import PPOTrainer

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

        # Compute log probs for current policy
        log_probs = F.log_softmax(lm_logits, dim=-1)
        labels_flat = input_ids.unsqueeze(-1)
        logprobs = torch.gather(log_probs, dim=-1, index=labels_flat).squeeze(-1)
        
    return logprobs, values

def load_dataset(file_path, data_fraction=1.0, accelerator=None):
    """
    Load dataset from a JSON file, filter based on data_fraction.
    
    Args:
        file_path: Path to the JSON dataset file
        data_fraction: Fraction of data to keep (0.0-1.0)
        accelerator: Optional accelerator for printing
    
    Returns:
        List of (question, value) tuples
    """
    # Load the raw data
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Extract question, answer_value pairs
    dataset = []
    for item in data:
        q = item["question"]
        val = float(item["answer_value"])
        dataset.append((q, val))
    
    # Subsample if data_fraction < 1.0
    if data_fraction < 1.0:
        random.shuffle(dataset)
        keep_sz = int(len(dataset) * data_fraction)
        dataset = dataset[:keep_sz]
    
    # Log dataset size
    if accelerator:
        accelerator.print(f"Loaded {len(dataset)} training samples from {file_path}")
    
    return dataset

def setup_tokenizer(warm_start_model, models, accelerator=None):
    """
    Load and configure tokenizer with special tokens.
    
    Args:
        warm_start_model: Path to the model checkpoint
        models: List of models that need embedding resizing if tokens added
        accelerator: Optional accelerator for printing
    
    Returns:
        Configured tokenizer
    """
    if accelerator:
        accelerator.print("Loading tokenizer")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(warm_start_model)
    
    # Always set padding_side to 'left' for decoder-only models
    tokenizer.padding_side = "left"
    
    # Set pad token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Verify the special tokens exist
    expected_tokens = ["<|begin_cot|>", "<|end_cot|>"]
    
    if accelerator:
        accelerator.print("Verifying special tokens in tokenizer vocabulary:")
        for token in expected_tokens:
            if token not in tokenizer.get_vocab():
                accelerator.print(f"WARNING: Token '{token}' not found in vocabulary! Chain-of-thought parsing may fail.")
            else:
                token_id = tokenizer.convert_tokens_to_ids(token)
                accelerator.print(f"Token {token} found with ID: {token_id}")
    
    # Add any missing tokens
    missing_tokens = [t for t in expected_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if missing_tokens:
        if accelerator:
            accelerator.print(f"Adding {len(missing_tokens)} missing special tokens to the tokenizer")
        
        special_tokens = {"additional_special_tokens": missing_tokens}
        num_added = tokenizer.add_special_tokens(special_tokens)
        
        # Resize embeddings for all provided models
        for model in models:
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
            elif hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "resize_token_embeddings"):
                model.pretrained_model.resize_token_embeddings(len(tokenizer))
        
        if accelerator:
            accelerator.print("Token embeddings have been resized. Note: newly added token embeddings are random!")
    
    return tokenizer

def calculate_reward_statistics(reward_tensor, attn_mask_padded, kl_penalty, reward_deque):
    """
    Calculate reward statistics from a batch of results.
    
    Args:
        reward_tensor: Tensor containing rewards for each token [batch_size, seq_len]
        attn_mask_padded: Attention mask to determine valid sequence lengths [batch_size, seq_len]
        kl_penalty: Optional tensor containing KL penalties
        reward_deque: Deque for tracking rolling average of rewards
        
    Returns:
        tuple: (avg_reward, avg_kl, avg_pool_reward)
    """
    batch_sz = reward_tensor.size(0)
    last_rewards = []
    avg_kl = 0.0
    kl_count = 0
    
    # Extract the reward at the last token of each sequence
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
    
    return avg_reward, avg_kl, avg_pool_reward

def log_and_save_checkpoints(
    accelerator, 
    wandb_run,
    ep,
    current_global_step,
    avg_reward,
    avg_pool_reward,
    loss_dict,
    avg_kl,
    kl_coef,
    policy_value_model,
    tokenizer,
    policy_optimizer,
    value_optimizer,
    best_reward,
    last_save_step,
    save_interval,
    saved_checkpoints,
    max_checkpoints_to_keep,
    output_dir
):
    """
    Handle logging metrics, saving checkpoints, and cleaning up old checkpoints.
    
    Args:
        accelerator: Accelerator instance for distributed logging
        wandb_run: Weights & Biases run object
        ep: Current epoch
        current_global_step: Current training step
        avg_reward: Average reward for the current batch
        avg_pool_reward: Rolling average of rewards
        loss_dict: Dictionary containing loss values
        avg_kl: Average KL divergence
        kl_coef: KL coefficient
        policy_value_model: The policy model
        tokenizer: The tokenizer
        policy_optimizer: Optimizer for policy network
        value_optimizer: Optimizer for value network
        best_reward: Current best reward (will be updated if needed)
        last_save_step: Last step where a checkpoint was saved
        save_interval: Interval for periodic saving
        saved_checkpoints: List of saved checkpoint paths
        max_checkpoints_to_keep: Maximum number of checkpoints to keep
        output_dir: Directory for saving checkpoints
        
    Returns:
        tuple: (updated_best_reward, updated_last_save_step, updated_saved_checkpoints)
    """
    # Only proceed if this is the main process
    if not accelerator.is_main_process:
        return best_reward, last_save_step, saved_checkpoints
    
    # 1) Log metrics to wandb
    if wandb_run and loss_dict is not None:
        wandb_run.log({
            "train/epoch": ep,
            "train/step": current_global_step,
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
        }, step=current_global_step)
    
    # 2) Print KL info
    accelerator.print(f"  KL coefficient: {kl_coef:.6f}, Avg KL: {avg_kl:.6f}")
    if avg_kl > 0.05:  # Arbitrary threshold
        accelerator.print("  ⚠️ WARNING: High KL divergence detected!")
    
    # 3) Check if we should save a checkpoint
    save_by_interval = (current_global_step - last_save_step) >= save_interval
    save_by_reward = avg_pool_reward > best_reward
    
    if save_by_interval or save_by_reward:
        # Create checkpoint directory
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{current_global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Unwrap and save model
        unwrapped_model = accelerator.unwrap_model(policy_value_model)
        accelerator.print(f"Saving checkpoint to {checkpoint_dir}")
        unwrapped_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer states
        torch.save(policy_optimizer.state_dict(), os.path.join(checkpoint_dir, "policy_optimizer.pt"))
        torch.save(value_optimizer.state_dict(), os.path.join(checkpoint_dir, "value_optimizer.pt"))
        
        # Save training state
        torch.save({
            "global_step": current_global_step,
            "rolling_avg_reward": avg_pool_reward,
            "best_reward": best_reward,
            "epoch": ep,
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Update tracking variables
        last_save_step = current_global_step
        saved_checkpoints.append(checkpoint_dir)
        
        # 4) If this is the best model so far, create a symlink
        if save_by_reward:
            best_reward = avg_pool_reward
            best_link = os.path.join(output_dir, "checkpoint-best")
            if os.path.exists(best_link):
                if os.path.islink(best_link):
                    os.unlink(best_link)
                else:
                    shutil.rmtree(best_link)
            # Create relative symlink
            os.symlink(os.path.basename(checkpoint_dir), best_link)
            accelerator.print(f"New best model with reward {best_reward:.4f}")
        
        # 5) Cleanup old checkpoints
        if len(saved_checkpoints) > max_checkpoints_to_keep:
            # Get the real path of the best checkpoint
            best_ckpt_path = None
            if os.path.exists(os.path.join(output_dir, "checkpoint-best")):
                best_ckpt_path = os.path.realpath(os.path.join(output_dir, "checkpoint-best"))
            
            # Get checkpoints to remove (oldest first)
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
            
            # Update our saved_checkpoints list
            saved_checkpoints = saved_checkpoints[-max_checkpoints_to_keep:]
    
    return best_reward, last_save_step, saved_checkpoints

def log_batch_results(
    accelerator,
    wandb_run,
    current_global_step,
    parsing_success,
    avg_reward,
    avg_pool_reward,
    final_texts,
    b_q,
    b_a,
    pred_values,
    reward_tensor,
    attn_mask_padded,
    temperature=None  # Add temperature parameter
):
    """
    Log detailed batch results including statistics and examples to wandb.
    
    Args:
        accelerator: Accelerator instance for distributed logging
        wandb_run: Weights & Biases run object (if available)
        current_global_step: Current training step
        parsing_success: List of boolean values indicating successful parsing
        avg_reward: Average reward for the current batch
        avg_pool_reward: Rolling average of rewards
        final_texts: Generated text responses
        b_q: Batch of questions/prompts
        b_a: Batch of true answers/values
        pred_values: Predicted values from the model
        reward_tensor: Tensor containing rewards [batch_size, seq_len]
        attn_mask_padded: Attention mask to determine sequence lengths
        temperature: Current temperature value for generation (optional)
    """
    # Only log if this is the main process
    if not accelerator.is_main_process:
        return
        
    # Print summary of batch results
    success_rate = sum(parsing_success) / len(parsing_success) * 100
    accelerator.print(f"\nBatch Summary - Step {current_global_step}:")
    accelerator.print(f"  Parsing success rate: {success_rate:.1f}%")
    accelerator.print(f"  Avg reward: {avg_reward:.4f} (rolling avg: {avg_pool_reward:.4f})")
    
    if temperature is not None:
        accelerator.print(f"  Temperature: {temperature:.4f}")
    
    # Log to wandb if enabled
    if wandb_run:
        # Log temperature if provided
        if temperature is not None:
            wandb_run.log({"train/temperature": temperature}, step=current_global_step)
        
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
                "reward": last_token_reward,
                "parsing_success": parsing_success[i]
            }
            wandb_run.log({f"examples/example_{i}": example}, step=current_global_step)

###############################################################################
# 5) MAIN
###############################################################################
def main(args):
    """
    Main PPO training function.
    
    Args:
        args: Parsed command line arguments
    """
    global_step = 0
    reward_deque = deque(maxlen=100)
    best_reward = -float('inf')
    last_save_step = 0
    save_interval = 100  # Save every 100 steps
    saved_checkpoints = []  # List to track saved checkpoints
    max_checkpoints_to_keep = 2  # Keep only 2 latest checkpoints
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    prompt_manager = PromptManager()

    # set seed
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args)
            )
        
        # Setup output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save args for future reference
        args_file = os.path.join(args.output_dir, "training_args.json")
        with open(args_file, "w") as f:
            json.dump(vars(args), f, indent=2)
        
        accelerator.print(f"Arguments saved to {args_file}")

    # Load dataset with separate function
    dataset = load_dataset(
        file_path=args.train_file,
        data_fraction=args.data_fraction,
        accelerator=accelerator
    )

    accelerator.print("Loading reference model from the same checkpoint")

    # Create the new BaseModel and freeze it
    base_model = BaseModel(
        warm_start_model=args.warm_start_model,
        base_model_name=args.base_model_name,
        accelerator=accelerator
    )
    base_model.freeze()
    ref_base = base_model.model

    accelerator.print("Loading policy_value_model as TRL AutoModelForCausalLMWithValueHead")
    policy_wrapper = PolicyValueModelWrapper(
        warm_start_model=args.warm_start_model,
        base_model_name=args.base_model_name,
        accelerator=accelerator
    )
    policy_wrapper.freeze_except_last_n_layers(n=2)
    policy_value_model = policy_wrapper.model

    # Wrap for DDP
    policy_value_model, ref_base = accelerator.prepare(policy_value_model, ref_base)
    

    # Setup tokenizer with a separate function
    tokenizer = setup_tokenizer(
        warm_start_model=args.warm_start_model,
        models=[ref_base, policy_value_model],
        accelerator=accelerator
    )

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

    # Calculate total steps for the temperature scheduler
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    max_steps = args.n_epochs * steps_per_epoch

    ppo_config = {
        "clip_range": args.clip_range,
        "vf_coef": args.vf_coef,
        "gamma": 0.99,
        "lam": 0.95,
        "do_sample": args.do_sample,
        "kl_coef": args.kl_coef,
        "max_new_tokens": args.max_new_tokens
    }
    ppo_trainer = PPOTrainer(
        policy_model=policy_value_model,
        ref_model=ref_base,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        config=ppo_config,
        accelerator=accelerator
    )
    
    # Initialize temperature scheduler
    ppo_trainer.set_temperature(
        start_temp=args.start_temp,
        end_temp=args.end_temp,
        total_steps=max_steps
    )

    # 6) PPO
    for ep in range(args.n_epochs):
        random.shuffle(dataset)
        for step_i in range(steps_per_epoch):
            # Compute the current global step
            current_global_step = ep * steps_per_epoch + step_i
            
            # Prepare batch
            batch_slice = dataset[step_i * args.batch_size:(step_i + 1) * args.batch_size]
            if not batch_slice:
                break
            b_q, b_a = zip(*batch_slice)

            # Create a frozen copy of the current policy to serve as old policy
            old_policy = copy.deepcopy(policy_value_model).eval()

            # Call the improved rollout_step with additional parameters
            (
                input_ids_padded,
                attn_mask_padded,
                train_mask,
                advantages,
                returns,
                old_logprobs,
                old_values,
                avg_reward,
                avg_kl,
                avg_pool_reward
            ) = ppo_trainer.rollout_step(
                ref_base=ref_base,
                policy_value_model=old_policy,
                tokenizer=tokenizer,
                prompts=b_q,
                true_values=b_a,
                reward_deque=reward_deque,
                current_global_step=current_global_step,
                wandb_run=wandb.run if accelerator.is_main_process else None
            )

            # PPO epochs (multiple updates per batch)
            for _ in range(args.ppo_epochs):
                loss_dict = ppo_trainer.ppo_step(
                    model=policy_value_model,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    old_logprobs=old_logprobs,
                    old_values=old_values,
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
                
            # Update global step counter and temperature step
            global_step += 1
            ppo_trainer.update_step()

            accelerator.wait_for_everyone()

            # Handle logging and checkpoint saving
            best_reward, last_save_step, saved_checkpoints = log_and_save_checkpoints(
                accelerator=accelerator,
                wandb_run=wandb.run if accelerator.is_main_process else None,
                ep=ep,
                current_global_step=current_global_step,
                avg_reward=avg_reward,
                avg_pool_reward=avg_pool_reward,
                loss_dict=loss_dict,
                avg_kl=avg_kl,
                kl_coef=args.kl_coef,
                policy_value_model=policy_value_model,
                tokenizer=tokenizer,
                policy_optimizer=policy_optimizer,
                value_optimizer=value_optimizer,
                best_reward=best_reward,
                last_save_step=last_save_step,
                save_interval=save_interval,
                saved_checkpoints=saved_checkpoints,
                max_checkpoints_to_keep=max_checkpoints_to_keep,
                output_dir=args.output_dir
            )

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

if __name__ == "__main__":
    # Move argument parsing here
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm_start_model", type=str, default="warmed_up_model",
                        help="Path to your *fine-tuned* model checkpoint.")
    parser.add_argument("--base_model_name", type=str, default="gpt2",
                        help="Base model name, e.g., 'openai/gpt2' or 'meta-llama/Llama-3.2-3B-Instruct'")
    parser.add_argument("--train_file", type=str, default="data/train_data.json",)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate for the policy network (default: 5e-6)")
    parser.add_argument("--value_lr", type=float, default=1e-7,
                        help="Learning rate for the value head (default: 1e-6)")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=random.randint(1, 10000),
                        help="Random seed for reproducibility (default: random)")
    parser.add_argument("--data_fraction", type=float, default=0.08)
    parser.add_argument("--wandb_project", type=str, default="ppo_nutri_g3")
    parser.add_argument("--wandb_entity", type=str, default="nmysore-uc-santa-barbara")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--do_sample", action="store_true", default="True", help="Use sampling instead of greedy.")
    parser.add_argument("--ppo_epochs", type=int, default=3,
                        help="Number of PPO update epochs per batch")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range parameter")
    parser.add_argument("--output_dir", type=str, default="ppo_trained_model",
                        help="Where to store the final PPO model (so inference can load it).")
    parser.add_argument("--vf_coef", type=float, default=0.1,
                        help="Value function coefficient for PPO")
    parser.add_argument("--start_temp", type=float, default=1.4,
                        help="Starting temperature for temperature scheduling")
    parser.add_argument("--end_temp", type=float, default=0.8,
                        help="Ending temperature for temperature scheduling")
    args = parser.parse_args()
    
    # Call main with the parsed arguments
    main(args)