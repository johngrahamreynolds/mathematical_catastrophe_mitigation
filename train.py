"""
Author: John Graham Reynolds

Training loop for Flan-T5-Base finetuning on Math + NLI tasks.

Supports single-task and mixed-task training with per-task loss tracking.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from typing import Dict, Optional, Any, Tuple
from tqdm import tqdm

from config import FullConfig, ExperimentConfig, compute_training_steps
from data import MixedDataLoader, create_dataloaders, create_mixed_dataloader
from evaluate import quick_eval, full_eval
from utils import (
    BestModelTracker,
    TensorBoardLogger,
    set_seed,
    get_gpu_memory_info,
    format_time,
    save_results,
)


# Try to import Apex FusedAdam, fall back to PyTorch AdamW
try:
    from apex.optimizers import FusedAdam # apex not installed by default; development done on Apple silicon
    APEX_AVAILABLE = True
    print("Apex FusedAdam available")
except ImportError:
    APEX_AVAILABLE = False
    print("Apex not available, using PyTorch AdamW")


def create_optimizer(
    model: T5ForConditionalGeneration,
    config: FullConfig,
) -> torch.optim.Optimizer:
    """Create optimizer (FusedAdam if available, else AdamW)."""
    
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if APEX_AVAILABLE:
        optimizer = FusedAdam(
            params,
            lr=config.train.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.train.weight_decay,
            adam_w_mode=True,
            set_grad_none=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=config.train.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.train.weight_decay,
        )
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: FullConfig,
    total_steps: int,
    warmup_steps: int,
):
    """Create learning rate scheduler."""
    if config.train.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif config.train.scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.train.scheduler_type}")
    
    return scheduler


def compute_mixed_losses(
    model: T5ForConditionalGeneration,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[float], Optional[float]]:
    """
    Compute loss with optional per-task decomposition.
    
    Returns:
        Tuple of (total_loss, math_loss, nli_loss)
        math_loss and nli_loss are floats for logging (detached)
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    
    total_loss = outputs.loss
    
    # If task_mask is present, compute per-task losses for logging
    if "task_mask" in batch:
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        
        per_token_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
        ).view(batch["labels"].shape)
        
        # Per-example loss
        mask = (batch["labels"] != -100).float()
        per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        task_mask = batch["task_mask"]
        math_mask = (task_mask == 0)
        nli_mask = (task_mask == 1)
        
        math_loss = per_example_loss[math_mask].mean().item() if math_mask.any() else None
        nli_loss = per_example_loss[nli_mask].mean().item() if nli_mask.any() else None
    else:
        math_loss = None
        nli_loss = None
    
    return total_loss, math_loss, nli_loss


def train_epoch(
    model: T5ForConditionalGeneration,
    dataloader: MixedDataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: FullConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    logger: TensorBoardLogger,
    best_tracker: BestModelTracker,
    tokenizer: T5Tokenizer,
    eval_dataloaders: Dict[str, DataLoader],
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[int, Dict[str, float]]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (updated global_step, epoch_metrics)
    """
    model.train()
    
    epoch_loss = 0.0
    epoch_math_loss = 0.0
    epoch_nli_loss = 0.0
    num_math_batches = 0
    num_nli_batches = 0
    
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch}",
    )
    
    for step_in_epoch, batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass with AMP
        if config.train.use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                total_loss, math_loss, nli_loss = compute_mixed_losses(model, batch)
        else:
            total_loss, math_loss, nli_loss = compute_mixed_losses(model, batch)
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss / config.train.gradient_accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        # Gradient accumulation
        if (step_in_epoch + 1) % config.train.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.train.max_grad_norm,
            )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Print GPU memory after first step (memory usage stabilizes after first backward)
            if global_step == 1:
                mem_info = get_gpu_memory_info()
                if mem_info:
                    print(f"\n📊 GPU Memory after first step:")
                    print(f"   Total:     {mem_info['total_gb']:.2f} GB")
                    print(f"   Allocated: {mem_info['allocated_gb']:.2f} GB")
                    print(f"   Reserved:  {mem_info['reserved_gb']:.2f} GB")
                    print(f"   Peak:      {mem_info['max_allocated_gb']:.2f} GB")
                    print(f"   Available: {mem_info['total_gb'] - mem_info['reserved_gb']:.2f} GB\n")
            
            # Logging
            epoch_loss += total_loss.item()
            if math_loss is not None:
                epoch_math_loss += math_loss
                num_math_batches += 1
            if nli_loss is not None:
                epoch_nli_loss += nli_loss
                num_nli_batches += 1
            
            # Log to TensorBoard (simple add_scalar pattern)
            if global_step % config.train.log_every_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.writer.add_scalar("train/loss", total_loss.item(), global_step)
                if math_loss is not None:
                    logger.writer.add_scalar("train/loss_math", math_loss, global_step)
                if nli_loss is not None:
                    logger.writer.add_scalar("train/loss_nli", nli_loss, global_step)
                logger.writer.add_scalar("train/learning_rate", current_lr, global_step)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Quick evaluation
            if global_step % config.train.eval_every_steps == 0:
                eval_metrics = quick_eval(
                    model=model,
                    math_dataloader=eval_dataloaders.get("math_quick_eval"),
                    nli_dataloader=eval_dataloaders.get("nli_quick_eval"),
                    tokenizer=tokenizer,
                    device=device,
                    num_samples=config.data.quick_eval_samples,
                    use_amp=config.train.use_amp,
                    dtype=dtype,
                )
                
                # Log eval metrics to TensorBoard
                logger.writer.add_scalar("eval/math_acc", eval_metrics["math_acc"], global_step)
                logger.writer.add_scalar("eval/math_loss", eval_metrics["math_loss"], global_step)
                logger.writer.add_scalar("eval/nli_acc", eval_metrics["nli_acc"], global_step)
                logger.writer.add_scalar("eval/nli_loss", eval_metrics["nli_loss"], global_step)
                logger.flush()
                
                print(f"\n  Step {global_step} | "
                      f"Math Acc: {eval_metrics['math_acc']:.3f} | "
                      f"NLI Acc: {eval_metrics['nli_acc']:.3f}")
                
                # Update best model tracker
                best_tracker.update(model, tokenizer, eval_metrics, global_step)
                
                model.train()  # Back to training mode
    
    # Compute epoch averages
    num_steps = len(dataloader) // config.train.gradient_accumulation_steps
    epoch_metrics = {
        "avg_loss": epoch_loss / num_steps if num_steps > 0 else 0,
        "avg_math_loss": epoch_math_loss / num_math_batches if num_math_batches > 0 else 0,
        "avg_nli_loss": epoch_nli_loss / num_nli_batches if num_nli_batches > 0 else 0,
    }
    
    return global_step, epoch_metrics


def train(
    config: FullConfig,
    experiment: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Main training function for a single experiment.
    
    Returns:
        Dictionary with training results and final metrics
    """
    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment.name}")
    print(f"Description: {experiment.description}")
    print(f"{'='*60}\n")
    
    # Set seed
    set_seed(config.train.seed)
    
    # Set experiment in config
    config.experiment = experiment
    
    # Device setup
    device = torch.device(config.train.device)
    dtype = config.model.dtype
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {config.model.model_name}")
    model = T5ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        dtype=dtype,
    ).to(device)
    
    tokenizer = T5Tokenizer.from_pretrained(config.model.model_name)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        config=config.data,
        tokenizer=tokenizer,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )
    
    # Create mixed dataloader for training
    if experiment.math_ratio > 0 and experiment.nli_ratio > 0:
        # Mixed training
        train_dataloader = create_mixed_dataloader(
            math_dataloader=dataloaders["math_train"],
            nli_dataloader=dataloaders["nli_train"],
            math_ratio=experiment.math_ratio,
            nli_ratio=experiment.nli_ratio,
        )
    elif experiment.math_ratio > 0:
        # Math-only training
        train_dataloader = create_mixed_dataloader(
            math_dataloader=dataloaders["math_train"],
            nli_dataloader=None,
            math_ratio=1,
            nli_ratio=0,
        )
    elif experiment.nli_ratio > 0:
        # NLI-only training
        train_dataloader = create_mixed_dataloader(
            math_dataloader=None,
            nli_dataloader=dataloaders["nli_train"],
            math_ratio=0,
            nli_ratio=1,
        )
    else:
        # Baseline - no training
        print("Baseline experiment - skipping training")
        
        # Just evaluate
        metrics = full_eval(
            model=model,
            math_dataloader=dataloaders["math_val"],
            nli_dataloader=dataloaders["nli_val"],
            tokenizer=tokenizer,
            device=device,
            use_amp=config.train.use_amp,
            dtype=dtype,
        )
        
        return {
            "experiment": experiment.name,
            "final_metrics": metrics,
            "training_time": 0,
            "epochs_completed": 0,
        }
    
    # Compute training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.train.num_epochs
    warmup_steps = int(config.train.warmup_ratio * total_steps)
    
    print(f"\nTraining configuration:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Batch size: {config.train.batch_size}")
    print(f"  Gradient accumulation: {config.train.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.train.batch_size * config.train.gradient_accumulation_steps}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, total_steps, warmup_steps)
    
    # Create best model tracker and logger
    best_tracker = BestModelTracker(
        checkpoint_dir=config.train.checkpoint_dir,
        experiment_name=experiment.name,
        best_metric=experiment.best_metric,
    )
    
    logger = TensorBoardLogger(
        log_dir=config.train.log_dir,
        experiment_name=experiment.name,
    )
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    for epoch in range(config.train.num_epochs):
        print(f"\n--- Epoch {epoch}/{config.train.num_epochs - 1} ---")
        
        global_step, epoch_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            epoch=epoch,
            global_step=global_step,
            logger=logger,
            best_tracker=best_tracker,
            tokenizer=tokenizer,
            eval_dataloaders=dataloaders,
            dtype=dtype,
        )
        
        # Log epoch-level metrics to TensorBoard
        logger.writer.add_scalar("train/loss_epoch", epoch_metrics['avg_loss'], epoch)
        logger.writer.add_scalar("train/loss_math_epoch", epoch_metrics['avg_math_loss'], epoch)
        logger.writer.add_scalar("train/loss_nli_epoch", epoch_metrics['avg_nli_loss'], epoch)
        logger.flush()
        
        print(f"\nEpoch {epoch} complete:")
        print(f"  Avg Loss: {epoch_metrics['avg_loss']:.4f}")
        print(f"  Avg Math Loss: {epoch_metrics['avg_math_loss']:.4f}")
        print(f"  Avg NLI Loss: {epoch_metrics['avg_nli_loss']:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {format_time(training_time)}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = full_eval(
        model=model,
        math_dataloader=dataloaders["math_val"],
        nli_dataloader=dataloaders["nli_val"],
        tokenizer=tokenizer,
        device=device,
        use_amp=config.train.use_amp,
        dtype=dtype,
    )
    
    print(f"\nFinal Results:")
    print(f"  Math Accuracy: {final_metrics['math_acc']:.4f}")
    print(f"  NLI Accuracy: {final_metrics['nli_acc']:.4f}")
    
    # Log final metrics to TensorBoard
    logger.writer.add_scalar("final/math_acc", final_metrics["math_acc"], global_step)
    logger.writer.add_scalar("final/math_loss", final_metrics["math_loss"], global_step)
    logger.writer.add_scalar("final/nli_acc", final_metrics["nli_acc"], global_step)
    logger.writer.add_scalar("final/nli_loss", final_metrics["nli_loss"], global_step)
    logger.flush()
    
    # Save final model
    best_tracker.save_final(model, tokenizer, final_metrics)
    
    # Print best vs final comparison
    best_summary = best_tracker.get_summary()
    print(f"\nBest {best_summary['best_metric']}: {best_summary['best_value']:.4f} at step {best_summary['best_step']}")
    print(f"Final {best_summary['best_metric']}: {final_metrics[best_summary['best_metric']]:.4f}")
    
    # Close logger
    logger.close()
    
    # Save results
    results = {
        "experiment": experiment.name,
        "config": {
            "math_ratio": experiment.math_ratio,
            "nli_ratio": experiment.nli_ratio,
            "learning_rate": config.train.learning_rate,
            "batch_size": config.train.batch_size,
            "num_epochs": config.train.num_epochs,
        },
        "final_metrics": final_metrics,
        "best_model": best_summary,
        "training_time": training_time,
        "epochs_completed": config.train.num_epochs,
        "total_steps": global_step,
        "gpu_memory": get_gpu_memory_info(),
    }
    
    results_path = os.path.join(config.train.output_dir, f"{experiment.name}_results.json")
    save_results(results, results_path)
    
    return results


def run_baseline_evaluation(
    config: FullConfig,
) -> Dict[str, Any]:
    """
    Run baseline evaluation without any training.
    
    Returns:
        Dictionary with baseline metrics
    """
    print("\n" + "="*60)
    print("Running baseline evaluation (pretrained model)")
    print("="*60 + "\n")
    
    # Device setup
    device = torch.device(config.train.device)
    dtype = config.model.dtype
    
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        dtype=dtype,
    ).to(device)
    
    tokenizer = T5Tokenizer.from_pretrained(config.model.model_name)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        config=config.data,
        tokenizer=tokenizer,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )
    
    # Evaluate
    metrics = full_eval(
        model=model,
        math_dataloader=dataloaders["math_val"],
        nli_dataloader=dataloaders["nli_val"],
        tokenizer=tokenizer,
        device=device,
        use_amp=config.train.use_amp,
        dtype=dtype,
    )
    
    print(f"\nBaseline Results:")
    print(f"  Math Accuracy: {metrics['math_acc']:.4f}")
    print(f"  NLI Accuracy: {metrics['nli_acc']:.4f}")
    
    results = {
        "experiment": "baseline",
        "final_metrics": metrics,
        "training_time": 0,
        "epochs_completed": 0,
    }
    
    results_path = os.path.join(config.train.output_dir, "baseline_results.json")
    save_results(results, results_path)
    
    return results

