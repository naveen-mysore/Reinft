import re

import torch
from transformers import GPT2Config, AutoConfig
from trl import AutoModelForCausalLMWithValueHead


class PolicyValueModelWrapper:
    """
    A small class that detects GPT2 vs. LLaMA, loads the policy+value model via TRL,
    and exposes freeze methods (e.g., freeze all, freeze all but last n layers).
    """

    def __init__(self, warm_start_model, base_model_name, accelerator):
        self.accelerator = accelerator
        self.model = None
        self.model_type = None

        # Detect GPT2 vs. LLaMA
        if "gpt2" in base_model_name.lower():
            self.model_type = "gpt2"
            accelerator.print("Detected GPT2-based policy; loading GPT2 with ValueHead...")
            config = GPT2Config.from_pretrained(warm_start_model, use_cache=False)
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                warm_start_model,
                config=config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            )
        else:
            self.model_type = "llama"
            accelerator.print("Detected LLaMA-based policy; loading LLaMA with ValueHead...")
            config = AutoConfig.from_pretrained(warm_start_model, use_cache=False)
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                warm_start_model,
                config=config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            )

        # Move to device
        self.model.to(self.accelerator.device)

    def freeze_all(self):
        """Freeze all parameters in the policy+value model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_except_last_n_layers(self, n=2):
        """
        Freeze all layers in the Transformer except for the last `n`.
        This also leaves the value head unfrozen, so it can learn.
        """
        # Check if the underlying model has a 'pretrained_model' attribute with 'model.layers'
        if hasattr(self.model.pretrained_model, "model") and \
           hasattr(self.model.pretrained_model.model, "layers"):
            total_layers = len(self.model.pretrained_model.model.layers)
            self.accelerator.print(f"PolicyValueModel has {total_layers} layers, freezing all but last {n}")
        else:
            total_layers = 0
            self.accelerator.print("WARNING: Could not find transformer layers in policy_value_model structure")

        # Record how many were trainable before
        total_params_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Freeze the layers below total_layers - n
        for name, param in self.model.pretrained_model.named_parameters():
            match = re.search(r"model\.layers\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                # If it's from layers < total_layers - n, freeze
                if layer_idx < (total_layers - n):
                    param.requires_grad = False
            else:
                # Freeze embeddings + other modules outside the last n layers
                param.requires_grad = False

        # Keep the ValueHead fully trainable
        for name, param in self.model.named_parameters():
            if 'pretrained_model' not in name:
                param.requires_grad = True

        # Summarize how many are left trainable
        total_params_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f"Trainable parameters reduced from {total_params_before:,} to {total_params_after:,}")
        if total_params_before > 0:
            ratio = (total_params_after / total_params_before) * 100
        else:
            ratio = 0.0
        self.accelerator.print(f"Now training {ratio:.2f}% of parameters")

    # You can add more specialized freeze methods if needed
