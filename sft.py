# Standard library imports
import argparse
import json
import os
import random
import re

# Third-party imports
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
)

# Local imports
from src.prompt_manager import PromptManager
from src.data_manager import DataManager


###############################################################################
# 1) CUSTOM TRAINER TO DISABLE USE_CACHE
###############################################################################
class MyTrainer(Trainer):
    """
    Override compute_loss to disable use_cache at each forward pass.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            use_cache=False
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


###############################################################################
# 2) TRAINING MANAGER CLASS
###############################################################################
class TrainingManager:
    """Class to manage the training process for the model."""

    def __init__(self, args, prompt_manager):
        """
        Initialize the training manager.

        Args:
            args: Command line arguments
            prompt_manager: Instance of PromptManager class
        """
        self.args = args
        self.prompt_manager = prompt_manager
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.trainer = None

    def _setup_device(self):
        """Set up the device for training."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device

    def setup_model_and_tokenizer(self):
        """Load and set up the model and tokenizer."""
        # Load config
        print(f"Loading config for {self.args.base_model_name}...")
        config = AutoConfig.from_pretrained(self.args.base_model_name)
        config.use_cache = False  # Disable caching

        # Load model
        print(f"Loading model {self.args.base_model_name}...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        if "gpt2" in self.args.base_model_name.lower():
            self.model = GPT2LMHeadModel.from_pretrained(
                self.args.base_model_name,
                config=config,
                torch_dtype=dtype,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.args.base_model_name,
                config=config,
                torch_dtype=dtype,
            )

        self.model.config.use_cache = False
        self.model.to(self.device)

        # Print parameter counts
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total model params: {param_count}, Trainable: {trainable_count}")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_name, use_fast=False)

        # Instead of hardcoding special_tokens, fetch them from the prompt manager:
        special_tokens = self.prompt_manager.get_model_special_tokens(self.args.base_model_name)
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to the tokenizer vocabulary")

        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        # Ensure pad token and EOS token are defined
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if self.tokenizer.eos_token_id is None:
            # fallback
            self.tokenizer.eos_token_id = self.tokenizer.pad_token_id

        print(f"EOS token: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")

        return self.model, self.tokenizer

    def setup_trainer(self):
        """Set up the trainer with the model and datasets."""
        if not self.model or not self.tokenizer or not self.train_dataset or not self.test_dataset:
            raise ValueError("Model, tokenizer, and datasets must be set up before creating the trainer")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(self.args.output_dir, "logs"),
            logging_steps=100,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported(),
            report_to=["wandb"],  # Always report to wandb
            run_name=self.args.wandb_run_name if self.args.wandb_run_name else "SFT-run",
            ddp_find_unused_parameters=False
        )

        # Create trainer
        self.trainer = MyTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=data_collator
        )

        # Make sure the model config has proper knowledge of EOS
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        return self.trainer

    def train(self):
        """Train the model."""
        if not self.trainer:
            raise ValueError("Trainer must be set up before training")

        print("Starting fine-tuning...")
        self.trainer.train()
        print("Fine-tuning complete!")

    def save_model(self):
        """Save the model and tokenizer."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be set up before saving")

        print("Saving final model to:", self.args.output_dir)
        self.trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print("All done! Check output in:", self.args.output_dir)


###############################################################################
# 3) HELPER FUNCTIONS
###############################################################################
def parse_args():
    # meta-llama/Llama-3.2-3B-Instruct
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train_data.json")
    parser.add_argument("--test_file", type=str, default="data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="warmed_up_model")
    parser.add_argument("--base_model_name", type=str, default="gpt2",
                        help="Base model name, e.g., 'openai/gpt2' or 'meta-llama/Llama-3.2-3B-Instruct'")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--data_fraction", type=float, default=0.001)
    parser.add_argument("--wandb_project", type=str, default="ppo_nutri_g3")
    parser.add_argument("--wandb_entity", type=str, default="nmysore-uc-santa-barbara")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


###############################################################################
# 4) MAIN TRAINING LOGIC
###############################################################################
def main():
    args = parse_args()

    # Initialize wandb (required)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args)
    )

    # Use a random seed instead of fixed seed
    random.seed()  # Using no argument makes Python choose a seed based on system time

    # Initialize managers
    prompt_manager = PromptManager()
    training_manager = TrainingManager(args, prompt_manager)
    model, tokenizer = training_manager.setup_model_and_tokenizer()

    # Create DataManager and load data
    data_manager = DataManager(prompt_manager, tokenizer)
    train_dataset, test_dataset = data_manager.load_and_process_datasets(
        train_file=args.train_file,
        test_file=args.test_file,
        data_fraction=args.data_fraction
    )
    # Assign them into training_manager if needed
    training_manager.train_dataset = train_dataset
    training_manager.test_dataset = test_dataset

    training_manager.setup_trainer()
    training_manager.train()
    training_manager.save_model()

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()