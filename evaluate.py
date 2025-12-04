"""
Author: John Graham Reynolds

Evaluation utilities for Math + NLI tasks.

Uses model.generate() for autoregressive decoding.
"""

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Optional
from tqdm import tqdm


# NLI output normalization - different model sizes output different phrasings
NLI_NEUTRAL_VARIANTS = {
    "it is not possible to tell",
    "it's impossible to say",
    "it's not possible to tell",
    "impossible to say",
    "it is impossible to say",
    "cannot be determined",
    "inconclusive",
}

def normalize_nli_output(text: str) -> str:
    """
    Normalize NLI outputs to handle different phrasings across model sizes.
    Maps various 'neutral' phrasings to a canonical form.
    """
    text = text.strip().lower()
    
    # Map neutral variants to canonical "neutral"
    if text in NLI_NEUTRAL_VARIANTS:
        return "neutral"
    
    # Return as-is (handles "yes", "no", math answers, etc.)
    return text


def evaluate(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_new_tokens: int = 8,
    max_examples: Optional[int] = None,
    desc: str = "Evaluating",
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """
    Evaluate model using generate() for autoregressive decoding.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation examples
        tokenizer: Tokenizer for decoding
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate (should match max_label_length)
        max_examples: Maximum examples to evaluate (for quick eval)
        desc: Description for progress bar
        use_amp: Whether to use automatic mixed precision
        dtype: Data type for AMP
    
    Returns:
        Dictionary with accuracy and loss metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            # Check if we've reached max examples
            if max_examples is not None and total >= max_examples:
                break
            
            # Move batch to device (exclude task_mask if present)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass for loss computation
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            # Accumulate loss
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Generate predictions (batched)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy decoding
                        num_beams=1,      # No beam search
                    )
            else:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            
            # Decode predictions and labels
            pred_strings = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            # For labels, replace -100 with pad_token_id before decoding
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            label_strings = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            # String comparison with NLI normalization
            for pred, label in zip(pred_strings, label_strings):
                pred_norm = normalize_nli_output(pred)
                label_norm = normalize_nli_output(label)
                if pred_norm == label_norm:
                    correct += 1
                total += 1
                
                # Debug: print first few predictions
                if total <= 5:
                    print(f"  [Debug] Pred: '{pred.strip().lower()}' → '{pred_norm}' | Label: '{label.strip().lower()}' → '{label_norm}' | Match: {pred_norm == label_norm}")
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "correct": correct,
        "total": total,
    }


def evaluate_both_tasks(
    model: T5ForConditionalGeneration,
    math_dataloader: Optional[DataLoader],
    nli_dataloader: Optional[DataLoader],
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_new_tokens: int = 8,
    max_examples: Optional[int] = None,
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on both Math and NLI tasks.
    
    Returns:
        Dictionary with results for each task
    """
    results = {}
    
    # Evaluate Math
    if math_dataloader is not None:
        math_results = evaluate(
            model=model,
            dataloader=math_dataloader,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            max_examples=max_examples,
            desc="Eval Math",
            use_amp=use_amp,
            dtype=dtype,
        )
        results["math"] = math_results
    
    # Evaluate NLI
    if nli_dataloader is not None:
        nli_results = evaluate(
            model=model,
            dataloader=nli_dataloader,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            max_examples=max_examples,
            desc="Eval NLI",
            use_amp=use_amp,
            dtype=dtype,
        )
        results["nli"] = nli_results
    
    return results


def quick_eval(
    model: T5ForConditionalGeneration,
    math_dataloader: Optional[DataLoader],
    nli_dataloader: Optional[DataLoader],
    tokenizer: T5Tokenizer,
    device: torch.device,
    num_samples: int = 1000,
    max_new_tokens: int = 8,
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """
    Quick evaluation on a subset for monitoring during training.
    
    Returns:
        Dictionary with accuracy and loss for each task
    """
    results = evaluate_both_tasks(
        model=model,
        math_dataloader=math_dataloader,
        nli_dataloader=nli_dataloader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        max_examples=num_samples,
        use_amp=use_amp,
        dtype=dtype,
    )
    
    return {
        "math_acc": results.get("math", {}).get("accuracy", 0.0),
        "math_loss": results.get("math", {}).get("loss", 0.0),
        "nli_acc": results.get("nli", {}).get("accuracy", 0.0),
        "nli_loss": results.get("nli", {}).get("loss", 0.0),
    }


def full_eval(
    model: T5ForConditionalGeneration,
    math_dataloader: Optional[DataLoader],
    nli_dataloader: Optional[DataLoader],
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_new_tokens: int = 8,
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """
    Full evaluation on complete validation sets.
    
    Returns:
        Dictionary with accuracy and loss for each task
    """
    results = evaluate_both_tasks(
        model=model,
        math_dataloader=math_dataloader,
        nli_dataloader=nli_dataloader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        max_examples=None,  # Full evaluation
        use_amp=use_amp,
        dtype=dtype,
    )
    
    return {
        "math_acc": results.get("math", {}).get("accuracy", 0.0),
        "math_loss": results.get("math", {}).get("loss", 0.0),
        "nli_acc": results.get("nli", {}).get("accuracy", 0.0),
        "nli_loss": results.get("nli", {}).get("loss", 0.0),
    }
