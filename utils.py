"""
Author: John Graham Reynolds

Utility functions for checkpointing and logging.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.tensorboard import SummaryWriter


def save_model(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    save_dir: str,
    metrics: Dict[str, float],
):
    """Save model, tokenizer, and metrics to directory."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved model to {save_path}")


class BestModelTracker:
    """
    Simple tracker for best model during training.
    Saves best + final checkpoints only.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        best_metric: str = "math_acc",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.best_metric = best_metric
        
        self.best_value = None
        self.best_step = None
    
    def update(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        metrics: Dict[str, float],
        step: int,
    ) -> bool:
        """Check if this is the best model so far. If so, save it."""
        current_value = metrics.get(self.best_metric, 0.0)
        
        is_best = (self.best_value is None or current_value > self.best_value)
        
        if is_best:
            self.best_value = current_value
            self.best_step = step
            
            save_dir = self.checkpoint_dir / f"{self.experiment_name}-best"
            save_model(model, tokenizer, str(save_dir), metrics)
            print(f"  New best {self.best_metric}: {current_value:.4f} at step {step}")
        
        return is_best
    
    def save_final(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        metrics: Dict[str, float],
    ):
        """Save final model."""
        save_dir = self.checkpoint_dir / f"{self.experiment_name}-final"
        save_model(model, tokenizer, str(save_dir), metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of best model."""
        return {
            "best_metric": self.best_metric,
            "best_value": self.best_value,
            "best_step": self.best_step,
        }


class TensorBoardLogger:
    """
    Simple TensorBoard logging wrapper.
    
    Usage:
        logger.writer.add_scalar("train/loss", loss.item(), global_step)
        logger.flush()
    """
    
    def __init__(self, log_dir: str, experiment_name: str = ""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name:
            exp_log_dir = self.log_dir / experiment_name
        else:
            exp_log_dir = self.log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.writer = SummaryWriter(log_dir=str(exp_log_dir))
        self.experiment_name = experiment_name
    
    def flush(self):
        """Flush writer to ensure data is written to disk."""
        self.writer.flush()
    
    def close(self):
        """Close the writer."""
        self.writer.close()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return {}
    
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024 / 1024 / 1024
    
    return {
        "total_gb": total_gb,
        "allocated_gb": torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        "reserved_gb": torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def save_results(results: Dict[str, Any], output_path: str):
    """Save experiment results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved results to {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """Load experiment results from JSON."""
    input_path = Path(input_path)
    if not input_path.exists():
        return {}
    
    with open(input_path, "r") as f:
        return json.load(f)
