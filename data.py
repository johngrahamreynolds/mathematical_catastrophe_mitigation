"""
Author: John Graham Reynolds

Data loading and preprocessing for Math + NLI training.

This module provides unified dataset classes and mixed dataloaders
for training on DeepMind Mathematics and MultiNLI datasets.

Note: The DeepMind Mathematics dataset must be loaded from pre-tokenized
versions on HuggingFace (the official dataset is deprecated).
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import Dict, Optional, Iterator, List
from itertools import cycle
import random

from config import DataConfig


class PretokenizedMathDataset(Dataset):
    """
    Dataset for pre-tokenized math data that extends padding to match NLI lengths.
    
    The pre-tokenized data has input_ids padded to a maximum of 40 tokens and labels padded to a maximum of 4 tokens.
    This class extends padding to max_input_length and max_label_length.
    """
    
    def __init__(
        self,
        hf_dataset,
        max_input_length: int = 128,
        max_label_length: int = 8,
        pad_token_id: int = 0,
    ):
        self.dataset = hf_dataset
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length
        self.pad_token_id = pad_token_id
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        return self._extend_padding(example)
    
    def _extend_padding(self, example: Dict) -> Dict[str, torch.Tensor]:
        """Extend padding of pre-tokenized example to target lengths."""
        # Get original tensors
        input_ids = list(example["input_ids"])
        attention_mask = list(example["attention_mask"])
        labels = list(example["labels"])
        
        # Current lengths
        curr_input_len = len(input_ids)
        curr_label_len = len(labels)
        
        # Extend input_ids with pad tokens
        if curr_input_len < self.max_input_length:
            pad_len = self.max_input_length - curr_input_len
            input_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        elif curr_input_len > self.max_input_length:
            # Truncate (shouldn't happen with math data, but just in case)
            input_ids = input_ids[:self.max_input_length]
            attention_mask = attention_mask[:self.max_input_length]
        
        # Extend labels with -100 (ignore index)
        if curr_label_len < self.max_label_length:
            pad_len = self.max_label_length - curr_label_len
            labels.extend([-100] * pad_len)
        elif curr_label_len > self.max_label_length:
            labels = labels[:self.max_label_length]
        
        # Convert -100 values in original labels (if pad_token was used instead)
        # and ensure pad tokens become -100
        labels = [
            -100 if (l == self.pad_token_id or l == -100) else l
            for l in labels
        ]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class UnifiedNLIDataset(Dataset):
    """
    Dataset class for NLI task that tokenizes on-the-fly.
    All examples are preprocessed to have identical tensor shapes.
    """
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: T5Tokenizer,
        max_input_length: int = 128,
        max_label_length: int = 8,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length
        
        # NLI label mapping (normalized - evaluation handles model output variants)
        self.nli_label_map = {0: "yes", 1: "neutral", 2: "no"}
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        return self._preprocess(example)
    
    def _preprocess(self, example: Dict) -> Dict[str, torch.Tensor]:
        """Preprocess a single NLI example."""
        # Flan-T5 format for NLI (premise first, then hypothesis)
        input_text = f"mnli premise: {example['premise']} hypothesis: {example['hypothesis']}"
        
        # Convert numeric label to text
        if isinstance(example["label"], int):
            label_text = self.nli_label_map[example["label"]]
        else:
            label_text = example["label"]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize label
        label_encoding = self.tokenizer(
            label_text,
            max_length=self.max_label_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Replace pad tokens in labels with -100 (ignored by loss)
        labels = label_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


class MixedDataLoader:
    """
    Mixed dataloader that yields batches from both Math and NLI datasets
    according to specified ratios.
    """
    
    def __init__(
        self,
        math_dataloader: Optional[DataLoader],
        nli_dataloader: Optional[DataLoader],
        math_ratio: int = 1,
        nli_ratio: int = 1,
        steps_per_epoch: Optional[int] = None,
    ):
        """
        Args:
            math_dataloader: DataLoader for math examples
            nli_dataloader: DataLoader for NLI examples
            math_ratio: Number of math batches per step
            nli_ratio: Number of NLI batches per step
            steps_per_epoch: Fixed number of steps per epoch (optional)
        """
        self.math_dataloader = math_dataloader
        self.nli_dataloader = nli_dataloader
        self.math_ratio = math_ratio
        self.nli_ratio = nli_ratio
        self.steps_per_epoch = steps_per_epoch
        
        # Create infinite iterators
        self._math_iter = None
        self._nli_iter = None
        self._reset_iterators()
        
        # Compute steps per epoch if not provided
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self._compute_steps_per_epoch()
    
    def _reset_iterators(self):
        """Reset the infinite iterators."""
        if self.math_dataloader is not None:
            self._math_iter = cycle(self.math_dataloader)
        if self.nli_dataloader is not None:
            self._nli_iter = cycle(self.nli_dataloader)
    
    def _compute_steps_per_epoch(self) -> int:
        """Compute steps per epoch. Datasets are assumed to be matched in size."""
        if self.math_dataloader is not None:
            return len(self.math_dataloader) // max(self.math_ratio, 1)
        elif self.nli_dataloader is not None:
            return len(self.nli_dataloader) // max(self.nli_ratio, 1)
        else:
            raise ValueError("At least one dataloader must be provided")
    
    def __len__(self) -> int:
        return self.steps_per_epoch
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield mixed batches."""
        self._reset_iterators()
        
        for _ in range(self.steps_per_epoch):
            yield self._get_mixed_batch()
    
    def _get_mixed_batch(self) -> Dict[str, torch.Tensor]:
        """Get a single mixed batch."""
        batches = []
        task_labels = []
        
        # Get math batches
        for _ in range(self.math_ratio):
            if self._math_iter is not None:
                batch = next(self._math_iter)
                batches.append(batch)
                task_labels.extend([0] * batch["input_ids"].size(0))
        
        # Get NLI batches
        for _ in range(self.nli_ratio):
            if self._nli_iter is not None:
                batch = next(self._nli_iter)
                batches.append(batch)
                task_labels.extend([1] * batch["input_ids"].size(0))
        
        if not batches:
            raise RuntimeError("No batches to combine")
        
        # Concatenate all batches
        combined = {
            "input_ids": torch.cat([b["input_ids"] for b in batches], dim=0),
            "attention_mask": torch.cat([b["attention_mask"] for b in batches], dim=0),
            "labels": torch.cat([b["labels"] for b in batches], dim=0),
            "task_mask": torch.tensor(task_labels, dtype=torch.long),
        }
        
        return combined


def load_math_dataset(
    config: DataConfig,
    tokenizer: T5Tokenizer,
    split: str = "train",
    seed: int = 1,
) -> PretokenizedMathDataset:
    """
    Load pre-tokenized DeepMind Mathematics dataset.
    
    The data is loaded from HuggingFace and padding is extended to match NLI lengths.
    """
    print(f"Loading Math dataset ({split})...")
    
    # Select the correct dataset based on split
    if split == "train":
        dataset_name = config.math_train_dataset
        hf_split = "train"
    else:  # validation/test
        dataset_name = config.math_val_dataset
        hf_split = "test"
    
    # Load pre-tokenized dataset
    dataset = load_dataset(dataset_name, split=hf_split)
    
    # Subsample if needed (with fixed seed for reproducibility)
    if split == "train" and config.math_train_size is not None:
        if len(dataset) > config.math_train_size:
            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), config.math_train_size)
            dataset = dataset.select(indices)
            print(f"  Subsampled to {config.math_train_size:,} examples (seed={seed})")
    elif split != "train" and config.math_val_size is not None:
        if len(dataset) > config.math_val_size:
            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), config.math_val_size)
            dataset = dataset.select(indices)
    
    # TODO: remove print statements after debugging
    print(f"  Loaded {len(dataset):,} examples")
    print(f"  Original padding: input={len(dataset[0]['input_ids'])}, labels={len(dataset[0]['labels'])}")
    print(f"  Extending to: input={config.max_input_length}, labels={config.max_label_length}")
    
    return PretokenizedMathDataset(
        hf_dataset=dataset,
        max_input_length=config.max_input_length,
        max_label_length=config.max_label_length,
        pad_token_id=tokenizer.pad_token_id,
    )


def load_nli_dataset(
    config: DataConfig,
    tokenizer: T5Tokenizer,
    split: str = "train",
    seed: int = 1,
) -> UnifiedNLIDataset:
    """Load and preprocess the MultiNLI dataset."""
    print(f"Loading NLI dataset ({split})...")
    
    # Map split names
    hf_split = split
    if split == "validation":
        hf_split = "validation_matched"  # Use matched validation
    
    # Load from HuggingFace
    dataset = load_dataset(config.nli_dataset, split=hf_split)
    
    # Subsample if needed (with fixed seed for reproducibility)
    if split == "train" and config.nli_train_size is not None:
        if len(dataset) > config.nli_train_size:
            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), config.nli_train_size)
            dataset = dataset.select(indices)
            print(f"  Subsampled to {config.nli_train_size:,} examples (seed={seed})")
    elif split == "validation" and config.nli_val_size is not None:
        if len(dataset) > config.nli_val_size:
            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), config.nli_val_size)
            dataset = dataset.select(indices)
    
    print(f"  Loaded {len(dataset):,} examples")
    
    # Convert to list for UnifiedNLIDataset
    examples = [
        {
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "label": ex["label"],
        }
        for ex in dataset
    ]
    
    return UnifiedNLIDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_label_length=config.max_label_length,
    )


def create_dataloaders(
    config: DataConfig,
    tokenizer: T5Tokenizer,
    batch_size: int,
    num_workers: int = 0,  # Default 0 to avoid multiprocessing issues in Colab
    seed: int = 1,
) -> Dict[str, DataLoader]:
    """Create all dataloaders for training and evaluation."""
    
    dataloaders = {}
    
    # Training dataloaders
    math_train = load_math_dataset(config, tokenizer, split="train", seed=seed)
    nli_train = load_nli_dataset(config, tokenizer, split="train", seed=seed)
    
    dataloaders["math_train"] = DataLoader(
        math_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    dataloaders["nli_train"] = DataLoader(
        nli_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Validation dataloaders
    math_val = load_math_dataset(config, tokenizer, split="test", seed=seed)
    nli_val = load_nli_dataset(config, tokenizer, split="validation", seed=seed)
    
    dataloaders["math_val"] = DataLoader(
        math_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    dataloaders["nli_val"] = DataLoader(
        nli_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Quick eval dataloaders (subsampled for fast evaluation during training)
    if config.quick_eval_samples is not None:
        rng = random.Random(seed)
        math_quick_indices = rng.sample(
            range(len(math_val)), 
            min(config.quick_eval_samples, len(math_val))
        )
        nli_quick_indices = rng.sample(
            range(len(nli_val)), 
            min(config.quick_eval_samples, len(nli_val))
        )
        
        dataloaders["math_quick_eval"] = DataLoader(
            Subset(math_val, math_quick_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        dataloaders["nli_quick_eval"] = DataLoader(
            Subset(nli_val, nli_quick_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders


def create_mixed_dataloader(
    math_dataloader: Optional[DataLoader],
    nli_dataloader: Optional[DataLoader],
    math_ratio: int,
    nli_ratio: int,
    steps_per_epoch: Optional[int] = None,
) -> MixedDataLoader:
    """Create a mixed dataloader with specified ratios."""
    return MixedDataLoader(
        math_dataloader=math_dataloader,
        nli_dataloader=nli_dataloader,
        math_ratio=math_ratio,
        nli_ratio=nli_ratio,
        steps_per_epoch=steps_per_epoch,
    )
