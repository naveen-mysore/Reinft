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
)

# Local imports
from prompt_manager import PromptManager


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
        print("Loading LLaMA config...")
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        config.use_cache = False  # Disable caching

        # Load model
        print("Loading LlamaForCausalLM model...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
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
        print("Loading LLaMA tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_fast=False)

        # For Llama, these tokens are typically built-in:
        # <s>, </s>, [INST], [/INST]
        # If you want to keep chain-of-thought tokens, add them:
        special_tokens = {
            "additional_special_tokens": [
                "<|begin_cot|>", 
                "<|end_cot|>"
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to the tokenizer vocabulary")

        self.model.resize_token_embeddings(len(self.tokenizer))

        # Ensure pad token and EOS token are defined
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if self.tokenizer.eos_token_id is None:
            # fallback
            self.tokenizer.eos_token_id = self.tokenizer.pad_token_id

        print(f"EOS token: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")

        return self.model, self.tokenizer

    def load_and_process_datasets(self, train_file, test_file, data_fraction):
        """Load and preprocess the datasets."""
        print("Loading datasets...")

        def load_json_as_hf_dataset(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
            return Dataset.from_list(data_list)

        train_dataset = load_json_as_hf_dataset(train_file)
        test_dataset = load_json_as_hf_dataset(test_file)

        # Shuffle
        train_dataset = train_dataset.shuffle(seed=random.randint(1, 10000))
        test_dataset = test_dataset.shuffle(seed=random.randint(1, 10000))

        # Fraction of data
        if data_fraction < 1.0:
            train_sz = int(len(train_dataset) * data_fraction)
            test_sz = int(len(test_dataset) * data_fraction)
            train_dataset = train_dataset.select(range(train_sz))
            test_dataset = test_dataset.select(range(test_sz))
            print(f"Using fraction={data_fraction}, train={len(train_dataset)}, test={len(test_dataset)}")
        else:
            print(f"Full data usage: train={len(train_dataset)}, test={len(test_dataset)}")

        # Preprocessing
        def preprocess_function(batch):
            # Rename columns to expected ones if needed
            questions = batch.get("question", batch.get("query", []))
            cots = batch.get("answer_cot", batch.get("cot", [""] * len(questions)))
            values = batch.get("answer_value", batch.get("answer", []))

            out_texts = []
            for q, cot, ans in zip(questions, cots, values):
                text = self.prompt_manager.build_training_sample(
                    query=q,
                    cot=cot, 
                    answer_value=str(ans),
                    eos_token=self.tokenizer.eos_token
                )
                # Safety check: ensure ends with EOS
                if not text.endswith(self.tokenizer.eos_token):
                    text += self.tokenizer.eos_token
                    print("WARNING: Added missing EOS token to example")
                    
                out_texts.append(text)

            tokenized = self.tokenizer(
                out_texts,
                truncation=False,
                padding=False,
                add_special_tokens=False
            )
            
            # Verify EOS tokens are present in a sample of examples
            if random.random() < 0.01:  # Check ~1% of batches
                for i in range(min(1, len(tokenized['input_ids']))):
                    if tokenized['input_ids'][i][-1] != self.tokenizer.eos_token_id:
                        print(f"WARNING: Example doesn't end with EOS: {tokenized['input_ids'][i][-5:]}")
                        # Force add EOS token at the end if missing
                        tokenized['input_ids'][i].append(self.tokenizer.eos_token_id)
                        tokenized['attention_mask'][i].append(1)
                        
            return tokenized

        print("Preprocessing train dataset...")
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        print("Preprocessing test dataset...")
        test_dataset = test_dataset.map(preprocess_function, batched=True)

        # Keep only input_ids / attention_mask
        keep_cols = {"input_ids", "attention_mask"}
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
        test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Debug prints
        print("[DEBUG] Checking random training sample(s)...")
        if len(self.train_dataset) > 0:
            idx = random.randint(0, len(self.train_dataset) - 1)
            sample = self.train_dataset[idx]
            decoded = self.tokenizer.decode(sample["input_ids"][:512])
            print(f"Train sample idx={idx}, decode[:512]:\n{decoded}\n")
        else:
            print("WARNING: Train dataset is empty!")

        print("[DEBUG] Checking random test sample(s)...")
        if len(self.test_dataset) > 0:
            idx = random.randint(0, len(self.test_dataset) - 1)
            sample = self.test_dataset[idx]
            decoded = self.tokenizer.decode(sample["input_ids"][:512])
            print(f"Test sample idx={idx}, decode[:512]:\n{decoded}\n")
        else:
            print("WARNING: Test dataset is empty!")

        # Basic length stats
        if len(self.train_dataset) > 0:
            avg_len = sum(len(self.train_dataset[i]["input_ids"]) for i in range(len(self.train_dataset))) / len(self.train_dataset)
            print(f"[DEBUG] Average train sample length: {avg_len:.2f} tokens.")
        if len(self.test_dataset) > 0:
            avg_len = sum(len(self.test_dataset[i]["input_ids"]) for i in range(len(self.test_dataset))) / len(self.test_dataset)
            print(f"[DEBUG] Average test sample length: {avg_len:.2f} tokens.")

        return self.train_dataset, self.test_dataset

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

    def prepare_data(self, df):
        """
        Prepare training data from a dataframe.
        This method builds examples with proper EOS token handling.
        """
        examples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing examples"):
            # Get the various fields
            query = row.get('query', row.get('question', ''))
            cot = row.get('cot', row.get('answer_cot', ''))  # May not exist in all datasets
            answer = str(row.get('answer', row.get('answer_value', '')))
            
            # Build the training sample with explicit EOS token
            sample = self.prompt_manager.build_training_sample(
                query=query,
                cot=cot,
                answer_value=answer,
                eos_token=self.tokenizer.eos_token  # Make sure to pass the correct EOS token
            )
            
            # Safety check: ensure it ends with proper EOS token
            if not sample.endswith(self.tokenizer.eos_token):
                sample += self.tokenizer.eos_token
                print(f"WARNING: Added missing EOS token to example {idx}")
            
            examples.append(sample)
        
        return examples


###############################################################################
# 3) HELPER FUNCTIONS
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train_data.json")
    parser.add_argument("--test_file", type=str, default="data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="/data/nmysore/models/fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--data_fraction", type=float, default=1.0)
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

    # Set up model and tokenizer
    training_manager.setup_model_and_tokenizer()

    # Load and process datasets
    training_manager.load_and_process_datasets(
        train_file=args.train_file,
        test_file=args.test_file,
        data_fraction=args.data_fraction
    )

    # Set up trainer
    training_manager.setup_trainer()

    # Train the model
    training_manager.train()

    # Save the model
    training_manager.save_model()

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()