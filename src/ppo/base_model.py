import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig, LlamaForCausalLM


class BaseModel:
    """
    A small class that detects GPT2 vs. LLaMA, loads the model,
    and can freeze all parameters as needed.
    """

    def __init__(self, warm_start_model, base_model_name, accelerator):
        self.accelerator = accelerator
        self.config = None
        self.model = None

        # Detect GPT2 vs. LLaMA, load config
        if "gpt2" in base_model_name.lower():
            print("Detected GPT2 model; loading GPT2 reference model...")
            self.config = GPT2Config.from_pretrained(warm_start_model)
            self.config.use_cache = False
            self.model = GPT2LMHeadModel.from_pretrained(
                warm_start_model,
                config=self.config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            )
        else:
            print("Detected LLaMA model; loading Llama reference model...")
            self.config = AutoConfig.from_pretrained(warm_start_model)
            self.config.use_cache = False
            self.model = LlamaForCausalLM.from_pretrained(
                warm_start_model,
                config=self.config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            )

        # Move to device
        self.model.to(self.accelerator.device)

    def freeze(self):
        """Freeze all parameters so they won't be updated during training."""
        for param in self.model.parameters():
            param.requires_grad = False
