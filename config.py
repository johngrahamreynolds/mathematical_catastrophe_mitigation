"""
Author: John Graham Reynolds


Configuration for Flan-T5-Base finetuning on Math + NLI tasks.

This module contains all hyperparameters and experiment definitions for
investigating catastrophic forgetting during mathematical reasoning finetuning.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "google/flan-t5-base"
    dtype: torch.dtype = torch.bfloat16
    

@dataclass
class DataConfig:
    """Data configuration."""
    # Dataset sources
    # Math: Use pre-tokenized datasets from HuggingFace (raw DeepMind dataset is deprecated)
    math_train_dataset: str = "MarioBarbeque/DeepMind-LinAlg-1D-train"
    math_val_dataset: str = "MarioBarbeque/DeepMind-LinAlg-1D-eval"
    math_pretokenized: bool = True  # Flag to indicate pre-tokenized data
    nli_dataset: str = "multi_nli"
    
    # Tokenization / Padding (all sequences padded to these lengths)
    max_input_length: int = 128  # Covers 95th percentile of NLI while reducing memory usage
    max_label_length: int = 8    # Covers all labels
    
    # Subsampling
    math_train_size: int = 392_702  # Match filtered MNLI train size
    math_val_size: Optional[int] = None  # Use full 10k
    nli_train_size: Optional[int] = None  # Use full MNLI train (~393k)
    nli_val_size: Optional[int] = None  # Use full matched val (~10k)
    
    # Evaluation subsampling (for quick evals during training)
    quick_eval_samples: int = 1000


@dataclass
class TrainConfig:
    """Training configuration."""
    # Core hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine" or "linear"
    warmup_ratio: float = 0.06
    
    # Evaluation frequency
    eval_every_steps: int = 500
    log_every_steps: int = 50
    
    # Directories
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Hardware
    device: str = "cuda"
    use_amp: bool = True  # Automatic mixed precision
    
    # Reproducibility
    seed: int = 1


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    name: str
    math_ratio: int  # Number of math batches per step
    nli_ratio: int   # Number of NLI batches per step
    best_metric: str = "math_acc"  # Metric to track for "best" checkpoint
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            if self.math_ratio == 0:
                self.description = "NLI-only training"
            elif self.nli_ratio == 0:
                self.description = "Math-only training"
            else:
                self.description = f"Mixed training with {self.math_ratio}:{self.nli_ratio} Math:NLI ratio"
        # NLI-only uses nli_acc as best metric
        if self.math_ratio == 0 and self.nli_ratio > 0:
            self.best_metric = "nli_acc"


# Define all experiments
EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline",
        math_ratio=0,
        nli_ratio=0,
        description="Pretrained model baseline (no training)"
    ),
    ExperimentConfig(
        name="math-only",
        math_ratio=1,
        nli_ratio=0,
    ),
    ExperimentConfig(
        name="nli-only",
        math_ratio=0,
        nli_ratio=1,
    ),
    ExperimentConfig(
        name="mixed-1-1",
        math_ratio=1,
        nli_ratio=1,
    ),
    # Original ratios, non-power-of-2 and indivisible by 256
    ExperimentConfig(
        name="mixed-2-1",
        math_ratio=2,
        nli_ratio=1,
    ),
    ExperimentConfig(
        name="mixed-4-1",
        math_ratio=4,
        nli_ratio=1,
    ),
    # Power-of-2 ratios (cleaner batch sizes: 256/sum = batch_size)
    ExperimentConfig(
        name="mixed-3-1",
        math_ratio=3,
        nli_ratio=1,
        description="Mixed training 3:1 Math:NLI (sum=4, batch=64 for 256 effective)"
    ),
    ExperimentConfig(
        name="mixed-7-1",
        math_ratio=7,
        nli_ratio=1,
        description="Mixed training 7:1 Math:NLI (sum=8, batch=32 for 256 effective)"
    ),
    ExperimentConfig(
        name="mixed-15-1",
        math_ratio=15,
        nli_ratio=1,
        description="Mixed training 15:1 Math:NLI (sum=16, batch=16 for 256 effective)"
    ),
]


def get_experiment_by_name(name: str) -> ExperimentConfig:
    """Get experiment config by name."""
    for exp in EXPERIMENTS:
        if exp.name == name:
            return exp
    raise ValueError(f"Experiment '{name}' not found. Available: {[e.name for e in EXPERIMENTS]}")


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: Optional[ExperimentConfig] = None
    
    def __post_init__(self):
        # Create output directories
        import os
        os.makedirs(self.train.output_dir, exist_ok=True)
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.log_dir, exist_ok=True)


def get_default_config() -> FullConfig:
    """Get default configuration."""
    return FullConfig()


# Convenience function to compute derived values
def compute_training_steps(config: FullConfig, num_train_examples: int) -> Dict[str, int]:
    """Compute total steps, warmup steps, etc."""
    effective_batch_size = config.train.batch_size * config.train.gradient_accumulation_steps
    steps_per_epoch = num_train_examples // effective_batch_size
    total_steps = steps_per_epoch * config.train.num_epochs
    warmup_steps = int(config.train.warmup_ratio * total_steps)
    
    return {
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }

