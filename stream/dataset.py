import polars as pl
from datasets import Dataset, DatasetDict
from unsloth import get_chat_template

class UnslothDatasetBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_dataset(self, df: pl.DataFrame) -> Dataset:
        """
        Transform a Polars DataFrame with colums `instruction`, `input`, `output`
        to an Hugging face compatible Dataset using Unsloth.
        """
        data = []

        for row in df.iter_rows(named=True):
            try:
                instruction = row["instruction"].strip()
                input_text = (row["input"] or "").strip()
                response = row["output"].strip()

                data.append({
                    "conversations": [
                        # {"from": "system", "value": instruction},
                        {"from": "human", "value": input_text},
                        {"from": "gpt", "value": response},
                    ]
                })
            except Exception as e:
                print(f"❌ Ligne ignorée : {e}")
                continue

        return Dataset.from_list(data)

    def tokenize(self):
        """Apply an unloth chat template"""
        return get_chat_template(
            self.tokenizer,
            chat_template="chatml",
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            map_eos_token=True,
        )

    def formatting_prompts_func(self, examples):
        """Format prompt for a good training with unsloth"""
        tokenizer = self.tokenize()
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    def split_dataset(self, dataset: Dataset, seed: int = 42) -> DatasetDict:
        """
        Split dataset to 60% train, 20% validation, 20% test.

        args:
            dataset : datasets.Dataset
                Dataset to split.

            seed : int
                seed for the random split

        Returns:
            DatasetDict containing sub datasets : train, validation, test
        """
        split = dataset.train_test_split(test_size=0.4, seed=seed)
        train_dataset = split["train"]
        temp_dataset = split["test"]

        temp_split = temp_dataset.train_test_split(test_size=0.5, seed=seed)
        val_dataset = temp_split["train"]
        test_dataset = temp_split["test"]
        print("\n" + "=" * 50)
        print(f"[DATASET SPLIT] → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        print("=" * 50 + "\n")

        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
