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
from utils import log_and_save_checkpoints


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