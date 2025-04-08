import json
import random
from datasets import Dataset

class DataManager:
    """
    Handles data loading, shuffling, preprocessing, and selection.
    """

    def __init__(self, prompt_manager, tokenizer):
        """Initialize with the PromptManager and tokenizer."""
        self.prompt_manager = prompt_manager
        self.tokenizer = tokenizer

    def load_and_process_datasets(self, train_file, test_file, data_fraction):
        """
        Load and preprocess datasets from JSON, shuffle, subset by fraction, and tokenize.
        Returns (train_dataset, test_dataset).
        """
        print("Loading datasets...")

        # Helper to load JSON as Hugging Face dataset
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
                # Ensure ends with eos_token
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

            # Optional: verify EOS tokens in a percentage of examples
            return tokenized

        print("Preprocessing train dataset...")
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        print("Preprocessing test dataset...")
        test_dataset = test_dataset.map(preprocess_function, batched=True)

        # Keep only input_ids / attention_mask
        keep_cols = {"input_ids", "attention_mask"}
        train_dataset = train_dataset.remove_columns(
            [c for c in train_dataset.column_names if c not in keep_cols]
        )
        test_dataset = test_dataset.remove_columns(
            [c for c in test_dataset.column_names if c not in keep_cols]
        )

        return train_dataset, test_dataset

    def prepare_data(self, df):
        """
        Prepare training data from a dataframe, building properly formatted
        samples with chain-of-thought and final answer, if present.
        """
        from tqdm import tqdm
        examples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing examples"):
            # Get the various fields
            query = row.get('query', row.get('question', ''))
            cot = row.get('cot', row.get('answer_cot', ''))  # optional
            answer = str(row.get('answer', row.get('answer_value', '')))

            # Build the training sample with explicit eos token
            sample = self.prompt_manager.build_training_sample(
                query=query,
                cot=cot,
                answer_value=answer,
                eos_token=self.tokenizer.eos_token
            )

            # Ensure it ends with eos_token
            if not sample.endswith(self.tokenizer.eos_token):
                sample += self.tokenizer.eos_token
                print(f"WARNING: Added missing EOS token to example {idx}")

            examples.append(sample)
        
        return examples 