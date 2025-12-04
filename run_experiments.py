"""
Author: John Graham Reynolds

Main entry point for running experiments.

This script runs all experiments defined in config.py and produces
comparative analysis of catastrophic forgetting during math finetuning.

Usage:
    # Run all experiments
    python run_experiments.py
    
    # Run specific experiment
    python run_experiments.py --experiment math-only
    
    # Run with custom config
    python run_experiments.py --learning-rate 5e-5 --batch-size 32
    
    # Calibration run (quick test)
    python run_experiments.py --calibration
"""

import argparse
import os
import time
from typing import Dict, List, Any

import torch

from config import (
    FullConfig,
    ExperimentConfig,
    get_experiment_by_name,
    get_default_config,
)
from train import train, run_baseline_evaluation
from utils import set_seed, save_results, load_results, format_time


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Flan-T5 finetuning experiments for Math + NLI"
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run specific experiment by name (e.g., 'math-only', 'nli-only', 'mixed-1-1', 'mixed-3-1', 'mixed-7-1', 'mixed-15-1')",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline evaluation (no training)",
    )
    
    # Hyperparameters (override config)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    
    # Data configuration
    parser.add_argument("--math-train-size", type=int, default=None)
    parser.add_argument("--nli-train-size", type=int, default=None)
    
    # Calibration mode
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run quick calibration (10k examples, 1 epoch)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for results",
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> FullConfig:
    """Create configuration from command line arguments."""
    config = get_default_config()
    
    # Override with command line args
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.train.num_epochs = args.num_epochs
    if args.gradient_accumulation is not None:
        config.train.gradient_accumulation_steps = args.gradient_accumulation
    if args.math_train_size is not None:
        config.data.math_train_size = args.math_train_size
    if args.nli_train_size is not None:
        config.data.nli_train_size = args.nli_train_size
    
    config.train.output_dir = args.output_dir
    config.train.seed = args.seed
    config.train.device = args.device
    
    # Calibration mode overrides
    if args.calibration:
        config.data.math_train_size = 10_000
        config.data.nli_train_size = 10_000
        config.train.num_epochs = 1
        config.train.eval_every_steps = 100
        print("Calibration mode: Using 10k examples, 1 epoch")
    
    return config


def get_experiments_to_run(args) -> List[ExperimentConfig]:
    """Get experiment to run based on args."""
    if args.baseline_only:
        return []  # No training, just baseline evaluation
    elif args.experiment is not None:
        return [get_experiment_by_name(args.experiment)]
    else:
        raise ValueError(
            "Must specify --experiment NAME or --baseline-only.\n"
            "Available experiments: math-only, nli-only, mixed-1-1, mixed-2-1, mixed-4-1"
        )


def main():
    """Main entry point."""
    args = parse_args()
    
    print("="*60)
    print("Flan-T5 Math + NLI Finetuning Experiments")
    print("="*60)
    
    # Create config
    config = create_config_from_args(args)
    
    # Set seed
    set_seed(config.train.seed)
    
    # Create output directory
    os.makedirs(config.train.output_dir, exist_ok=True)
    
    # Load existing results (to append, not overwrite)
    all_results_path = os.path.join(config.train.output_dir, "all_results.json")
    all_results: Dict[str, Dict] = load_results(all_results_path)
    if all_results:
        print(f"Loaded existing results: {list(all_results.keys())}")
    
    # Run baseline evaluation first
    if not args.skip_baseline:
        print("\n" + "-"*40)
        print("Step 1: Baseline Evaluation")
        print("-"*40)
        baseline_results = run_baseline_evaluation(config)
        all_results["baseline"] = baseline_results
    
    # Get experiments to run
    experiments = get_experiments_to_run(args)
    
    if experiments:
        print(f"\nExperiments to run: {[e.name for e in experiments]}")
    elif args.baseline_only:
        print("\nBaseline-only mode: No training experiments will run.")
    else:
        print("\nNo experiments to run.")
    
    # Run experiments
    total_start_time = time.time()
    
    for i, experiment in enumerate(experiments):
        print(f"\n" + "-"*40)
        print(f"Step {i+2}: Experiment '{experiment.name}'")
        print("-"*40)
        
        results = train(config, experiment)
        all_results[experiment.name] = results
        
        # Save intermediate results
        save_results(
            all_results,
            os.path.join(config.train.output_dir, "all_results.json"),
        )
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nTotal time: {format_time(total_time)}")
    print(f"Results saved to: {config.train.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. View training curves: tensorboard --logdir={config.train.log_dir}")
    print(f"  2. Generate plots: python generate_plots.py --results-dir {config.train.output_dir}")
    print(f"  3. Push to Hub: python publish_to_hub.py --checkpoint-dir <path> --repo-name <name>")


if __name__ == "__main__":
    main()

