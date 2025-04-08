import torch
import torch.nn.functional as F
import numpy as np
import math
import wandb
from collections import deque


class PPOTrainer:
    def __init__(self, policy_model, ref_model, policy_optimizer, value_optimizer, config, accelerator=None):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.config = config
        self.accelerator = accelerator
        # Initialize temperature tracking
        self.start_temp = 1.0
        self.end_temp = 0.5
        self.total_steps = 1000  # Default value
        self.current_step = 0
        
        # Initialize prompt manager
        from src.prompt_manager import PromptManager
        self.prompt_manager = PromptManager()

    def set_temperature(self, start_temp, end_temp, total_steps):
        """
        Set up temperature annealing parameters.
        
        Args:
            start_temp: Starting temperature at step=0
            end_temp: Final temperature when step=total_steps
            total_steps: The maximum number of steps in your entire training
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = max(1, total_steps)  # Ensure not zero
        self.current_step = 0

    def get_current_temperature(self):
        """
        Get the current temperature based on linear schedule.
        Also increments the current_step counter.
        
        Returns:
            Current temperature value
        """
        # Cap the fraction between 0 and 1
        fraction = min(max(float(self.current_step) / float(self.total_steps), 0.0), 1.0)
        
        # Linear interpolation
        temp = self.start_temp + fraction * (self.end_temp - self.start_temp)
        return temp

    def update_step(self):
        """
        Increment the current step counter for temperature scheduling.
        """
        self.current_step += 1

    def rollout_step(
        self,
        ref_base,
        policy_value_model,
        tokenizer,
        prompts: list[str],
        true_values: list[float],
        reward_deque=None,
        current_global_step=0,
        wandb_run=None
    ):
        """
        Execute a complete rollout step, including computing advantages and returns,
        calculating statistics, and logging results.
        
        Args:
            ref_base: Reference model for KL penalty
            policy_value_model: Policy value model for forward pass
            tokenizer: Tokenizer for processing text
            prompts: List of prompts to process
            true_values: List of true values corresponding to prompts
            reward_deque: Optional deque for tracking rolling rewards
            current_global_step: Current training step for logging
            wandb_run: Optional wandb run object for logging
        
        Returns:
            tuple containing:
            - input_ids_padded: Padded input token IDs
            - attn_mask_padded: Attention mask for padding
            - train_mask: Mask for which tokens to train on
            - advantages: Normalized advantages
            - returns: Returns for each token
            - old_logprobs: Log probabilities from old policy
            - old_values: Value estimates from old policy
            - avg_reward: Average reward for the batch
            - avg_kl: Average KL divergence
            - avg_pool_reward: Rolling average reward
        """
        # Use self.accelerator instead of parameter
        accelerator = self.accelerator
        
        # Get parameters from config
        do_sample = self.config.get("do_sample", False)
        kl_coef = self.config.get("kl_coef", 0.02)
        max_new_tokens = self.config.get("max_new_tokens", 200)
        gamma = self.config.get("gamma", 0.99)
        lam = self.config.get("lam", 0.95)
        
        # Get current temperature from internal state
        temperature = self.get_current_temperature()
        
        device = next(policy_value_model.parameters()).device
        batch_size = len(prompts)

        # 1) Build prompts - use internal prompt_manager
        training_prompts = [self.prompt_manager.build_prompt(p, mode="inference") for p in prompts]
        enc = tokenizer(training_prompts, return_tensors="pt", padding=True, truncation=True)
        base_input_ids = enc["input_ids"].to(device)
        base_attention_mask = enc["attention_mask"].to(device)

        # 2) Generate text with unwrapped policy_value_model
        with torch.no_grad():
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

        # 4) Optional logging
        if accelerator and accelerator.is_main_process:
            num_examples = min(2, len(final_texts))
            accelerator.print("\n" + "="*80)
            accelerator.print(f"SAMPLE GENERATIONS (temp={temperature}, do_sample={do_sample}):")
            for i in range(num_examples):
                accelerator.print(f"\nEXAMPLE {i+1}:")
                accelerator.print(f"PROMPT: {prompts[i][:100]}...")
                accelerator.print(f"RESPONSE:\n{final_texts[i]}")
            accelerator.print("="*80 + "\n")

        # 5) Parse each response, compute numeric reward
        parsing_success = [False] * batch_size
        cots = [None] * batch_size
        pred_values = [None] * batch_size
        reward_list = [0.0] * batch_size
        
        for i in range(batch_size):
            text = final_texts[i]
            # Parse chain-of-thought from the text
            cot, ans_val = self.prompt_manager.parse_cot_and_answer(text)
            cots[i] = cot
            
            is_malformed = False
            try:
                guess = float(ans_val)
                parsing_success[i] = True
            except ValueError:
                guess = None
                parsing_success[i] = False
                is_malformed = True
                
            pred_values[i] = guess
            if guess is None or is_malformed:
                numeric_reward = -1.0
            else:
                true_val = float(true_values[i])
                numeric_reward = gaussian_reward(
                    guess,
                    true_val,
                    sigma=10.0,
                    near_exact_range=5.0,
                    near_exact_bonus=1.0
                )
            reward_list[i] = numeric_reward
            
            if accelerator and accelerator.is_main_process and i < 3:
                accelerator.print(f"  Final prediction: {guess}")
                accelerator.print(f"  True value: {true_values[i]}")
                accelerator.print(f"  Reward: {numeric_reward}")
                accelerator.print(f"  Is malformed: {is_malformed}")

        # 6) Build padded tensors
        batch_lens = [seq.shape[0] for seq in new_ids]
        max_seq_len = max(batch_lens)

        input_ids_padded = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        attn_mask_padded = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)
        train_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)
        reward_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)

        for b in range(batch_size):
            seq_len_b = batch_lens[b]
            if seq_len_b > 0:
                input_ids_padded[b, :seq_len_b] = new_ids[b]
                attn_mask_padded[b, :seq_len_b] = 1.0

                # Mark tokens for training (not including the prompt)
                prompt_tokens = tokenizer(training_prompts[b], return_tensors="pt").input_ids[0]
                prompt_len = min(len(prompt_tokens), seq_len_b)
                if prompt_len < seq_len_b:
                    train_mask[b, prompt_len:seq_len_b] = 1.0
                
                # Reward on last token
                reward_tensor[b, seq_len_b - 1] = reward_list[b]

        # 7) Calculate KL penalty if ref_base is provided
        kl_penalty = get_kl_penalty(
            policy_value_model=policy_value_model,
            ref_base=ref_base,
            input_ids_padded=input_ids_padded,
            attn_mask_padded=attn_mask_padded,
            batch_lens=batch_lens,
            train_mask=train_mask,
            kl_coef=kl_coef,
        )
        
        # Subtract KL from reward
        reward_tensor = reward_tensor - kl_penalty

        # Create a default reward_deque if not provided
        if reward_deque is None:
            reward_deque = deque(maxlen=100)
        
        # Calculate statistics
        avg_reward, avg_kl, avg_pool_reward = self.calculate_reward_statistics(
            reward_tensor=reward_tensor,
            attn_mask_padded=attn_mask_padded,
            kl_penalty=kl_penalty,
            reward_deque=reward_deque
        )
        
        # Log batch results
        self.log_batch_results(
            wandb_run=wandb_run,
            current_global_step=current_global_step,
            parsing_success=parsing_success,
            avg_reward=avg_reward,
            avg_pool_reward=avg_pool_reward,
            final_texts=final_texts,
            prompts=prompts,
            true_values=true_values,
            pred_values=pred_values,
            reward_tensor=reward_tensor,
            attn_mask_padded=attn_mask_padded
        )
        
        # Calculate log probs and values from the old policy
        old_logprobs, old_values = self.get_log_probs_and_values(
            policy=policy_value_model,
            input_ids=input_ids_padded,
            attention_mask=attn_mask_padded,
            use_cache=False
        )
        
        # Compute advantages & returns
        advantages, returns = self.compute_normalized_gae_advantages(
            rewards=reward_tensor,
            values=old_values,
            mask=attn_mask_padded * train_mask,
            gamma=gamma,
            lam=lam
        )
        
        # Return only what's needed for PPO update and reporting
        return (
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
        )

    def ppo_step(
        self,
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
        Mirrors the logic taken from reft.py's original ppo_step code.
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
        log_probs = F.log_softmax(lm_logits, dim=-1)
        labels_flat = input_ids.unsqueeze(-1)
        new_logprobs = torch.gather(log_probs, dim=-1, index=labels_flat).squeeze(-1)

        # Combine attention mask with train mask
        valid_mask = attn_mask.float() * train_mask

        # Check for empty mask
        if valid_mask.sum() < 1.0:
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
        ratio = torch.exp(delta_logprobs)
        
        # Compute policy loss
        adv_masked = advantages * valid_mask
        ratio_masked = ratio * valid_mask
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

        # Combine
        total_loss = policy_loss + vf_coef * value_loss
        
        # 1) Zero gradients
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        
        # 2) Backprop
        total_loss.backward()
        
        # 3) Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 4) Update
        policy_optimizer.step()
        value_optimizer.step()

        # Gather stats
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

    def compute_normalized_gae_advantages(self, rewards, values, gamma=0.99, lam=0.95, mask=None):
        """
        Compute Generalized Advantage Estimation across token sequences with normalization.
        
        Args:
            rewards: Tensor of shape [batch_size, seq_len] with rewards
            values: Tensor of shape [batch_size, seq_len] with value predictions
            gamma: Discount factor
            lam: GAE lambda parameter
            mask: Optional attention mask to identify valid tokens
        
        Returns:
            advantages: Tensor of shape [batch_size, seq_len], normalized to mean=0, std=1
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
        
        # Normalize advantages to have zero mean and unit variance
        with torch.no_grad():
            valid_mask = mask > 0
            if valid_mask.any():
                adv_valid = advantages[valid_mask]
                mean_adv = adv_valid.mean()
                std_adv = adv_valid.std().clamp_min(1e-8)
                # Normalize in place
                advantages = (advantages - mean_adv) / std_adv
        
        return advantages, returns

    def get_log_probs_and_values(self, policy, input_ids, attention_mask, use_cache=False):
        """
        Perform a forward pass with a policy model and extract log probabilities and values.
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

    def calculate_reward_statistics(self, reward_tensor, attn_mask_padded, kl_penalty, reward_deque):
        """
        Calculate reward statistics from a batch of results.
        
        Args:
            reward_tensor: Tensor containing rewards for each token [batch_size, seq_len]
            attn_mask_padded: Attention mask to determine valid sequence lengths
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

    def log_batch_results(
        self,
        wandb_run,
        current_global_step,
        parsing_success,
        avg_reward,
        avg_pool_reward,
        final_texts,
        prompts,
        true_values,
        pred_values,
        reward_tensor,
        attn_mask_padded
    ):
        """
        Log detailed batch results including statistics and examples to wandb.
        """
        # Use self.accelerator instead of passing it
        accelerator = self.accelerator
        
        # Only log if this is the main process
        if not accelerator or not accelerator.is_main_process:
            return
        
        # Print summary of batch results
        success_rate = sum(parsing_success) / len(parsing_success) * 100
        accelerator.print(f"\nBatch Summary - Step {current_global_step}:")
        accelerator.print(f"  Parsing success rate: {success_rate:.1f}%")
        accelerator.print(f"  Avg reward: {avg_reward:.4f} (rolling avg: {avg_pool_reward:.4f})")
        
        # Get temperature from our own method
        temperature = self.get_current_temperature()
        accelerator.print(f"  Temperature: {temperature:.4f}")
        
        # Log to wandb if enabled
        if wandb_run:
            # Log temperature
            wandb_run.log({"train/temperature": temperature}, step=current_global_step)
            
            # Log a few examples with their parsed values
            log_samples = min(3, len(final_texts))
            for i in range(log_samples):
                # Get reward from the last token of reward_tensor for this example
                seq_len = int(attn_mask_padded[i].sum().item())
                last_token_reward = reward_tensor[i, seq_len-1].item() if seq_len > 0 else 0.0
                
                example = {
                    "prompt": prompts[i],
                    "response": final_texts[i],
                    "parsed_value": pred_values[i],
                    "true_value": true_values[i],
                    "reward": last_token_reward,
                    "parsing_success": parsing_success[i]
                }
                wandb_run.log({f"examples/example_{i}": example}, step=current_global_step)

def get_kl_penalty(
    policy_value_model,
    ref_base,
    input_ids_padded,
    attn_mask_padded,
    batch_lens,
    train_mask,
    kl_coef=0.02,
):
    """
    Calculate KL divergence penalty between policy and reference model.
    Applies a uniform KL coefficient across all tokens.
    """
    if ref_base is None or kl_coef <= 0.0:
        return torch.zeros_like(input_ids_padded, dtype=torch.float)
    
    batch_size = input_ids_padded.size(0)
    max_seq_len = input_ids_padded.size(1)
    
    # Initialize KL penalty tensor
    kl_penalty = torch.zeros_like(input_ids_padded, dtype=torch.float)
    
    # Get logits from both models
    with torch.no_grad():
        # Policy model logits
        policy_out = policy_value_model(
            input_ids=input_ids_padded,
            attention_mask=attn_mask_padded,
            use_cache=False
        )
        if isinstance(policy_out, tuple) and len(policy_out) >= 2:
            policy_logits = policy_out[0]
        else:
            policy_logits = policy_out.logits
        
        # Reference model logits
        ref_out = ref_base(
            input_ids=input_ids_padded,
            attention_mask=attn_mask_padded,
            use_cache=False
        )
        if isinstance(ref_out, tuple) and len(ref_out) >= 2:
            ref_logits = ref_out[0]
        else:
            ref_logits = ref_out.logits
    
    # Calculate KL div token by token (only for tokens we train on)
    for b in range(batch_size):
        seq_len = batch_lens[b]
        
        for t in range(seq_len):
            # Only apply KL penalty to tokens we're training on
            if train_mask[b, t] == 0:
                continue
                
            # Get logits for this position
            p_logits = policy_logits[b, t]
            r_logits = ref_logits[b, t]
            
            # Calculate KL div
            p_log_softmax = F.log_softmax(p_logits, dim=-1)
            r_softmax = F.softmax(r_logits, dim=-1)
            token_kl = F.kl_div(p_log_softmax, r_softmax, reduction='sum')
            
            # Apply uniform KL coefficient
            kl_penalty[b, t] = kl_coef * token_kl
    
    return kl_penalty

def gaussian_reward(
        pred_value: float,
        true_value: float,
        sigma: float = 10.0,
        near_exact_range: float = 5.0,
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
    base_reward = 2 * math.exp(-(diff ** 2) / (2.0 * sigma * sigma))

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