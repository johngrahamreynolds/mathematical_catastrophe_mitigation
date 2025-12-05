"""
Generate plots from experiment results.

Usage:
    python generate_acc_plots.py
    python generate_acc_plots.py --results-dir ./outputs --output-dir ./outputs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all experiment results from JSON files."""
    results_dir = Path(results_dir)
    all_results = {}
    
    for json_file in results_dir.glob("*_results.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        if "experiment" in data:
            exp_name = data["experiment"]
            all_results[exp_name] = data
        else:
            all_results.update(data)
    
    print(f"Loaded {len(all_results)} experiment results:")
    for name in sorted(all_results.keys()):
        print(f"  - {name}")
    
    return all_results


def plot_accuracy_comparison(all_results: Dict[str, Dict], output_dir: Path):
    """Bar chart comparing Math and NLI accuracy across experiments."""
    experiments = []
    math_accs = []
    nli_accs = []
    
    order = ["baseline", "math-only", "nli-only", "mixed-1-1", "mixed-3-1", "mixed-7-1", "mixed-15-1"]
    for exp_name in order:
        if exp_name in all_results and "final_metrics" in all_results[exp_name]:
            experiments.append(exp_name)
            math_accs.append(all_results[exp_name]["final_metrics"].get("math_acc", 0) * 100)
            nli_accs.append(all_results[exp_name]["final_metrics"].get("nli_acc", 0) * 100)
    
    if len(experiments) < 2:
        print("Need at least 2 experiments for comparison plot. Skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(experiments))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, math_accs, width, label='Math Accuracy', color='#2ecc71')
    bars2 = ax.bar(x + width/2, nli_accs, width, label='NLI Accuracy', color='#3498db')
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Math vs NLI Accuracy Across Training Regimes', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")


def plot_pareto_frontier(all_results: Dict[str, Dict], output_dir: Path):
    """Scatter plot showing Math vs NLI accuracy tradeoff."""
    experiments = []
    math_accs = []
    nli_accs = []
    
    # Define a logical order for experiments to ensure consistent legend ordering
    exp_order = ["baseline", "math-only", "nli-only", "mixed-1-1",
                 "mixed-3-1", "mixed-7-1", "mixed-15-1"]
    
    # Collect data in the defined order
    for exp_name in exp_order:
        if exp_name in all_results and "final_metrics" in all_results[exp_name]:
            experiments.append(exp_name)
            math_accs.append(all_results[exp_name]["final_metrics"].get("math_acc", 0) * 100)
            nli_accs.append(all_results[exp_name]["final_metrics"].get("nli_acc", 0) * 100)
    
    if len(experiments) < 2:
        print("Need at least 2 experiments for Pareto plot. Skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # Plot points with labels for legend (no annotations on plot)
    for i, (exp, math_acc, nli_acc) in enumerate(zip(experiments, math_accs, nli_accs)):
        ax.scatter(nli_acc, math_acc, c=[colors[i]], s=200, label=exp, zorder=5,
                   edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('NLI Accuracy (%)', fontsize=12)
    ax.set_ylabel('Math Accuracy (%)', fontsize=12)
    ax.set_title('Pareto Frontier: Math vs NLI Performance', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(max(0, min(nli_accs) - 10), min(100, max(nli_accs) + 10))
    ax.set_ylim(max(0, min(math_accs) - 10), min(100, max(math_accs) + 10))
    
    # Add legend inside the plot area (upper right corner to avoid data overlap)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_frontier.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'pareto_frontier.png'}")


def plot_performance_change(all_results: Dict[str, Dict], output_dir: Path):
    """Bar chart showing change from baseline."""
    if "baseline" not in all_results:
        print("No baseline results found. Skipping performance change plot.")
        return
    
    baseline_math = all_results["baseline"]["final_metrics"].get("math_acc", 0) * 100
    baseline_nli = all_results["baseline"]["final_metrics"].get("nli_acc", 0) * 100
    
    order = ["math-only", "nli-only", "mixed-1-1", "mixed-3-1", "mixed-7-1", "mixed-15-1"]
    exp_names = []
    math_changes = []
    nli_changes = []
    
    for exp in order:
        if exp in all_results and "final_metrics" in all_results[exp]:
            exp_names.append(exp)
            math_acc = all_results[exp]["final_metrics"].get("math_acc", 0) * 100
            nli_acc = all_results[exp]["final_metrics"].get("nli_acc", 0) * 100
            math_changes.append(math_acc - baseline_math)
            nli_changes.append(nli_acc - baseline_nli)
    
    if not exp_names:
        print("No training experiments found. Skipping performance change plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(exp_names))
    width = 0.35
    
    math_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in math_changes]
    nli_colors = ['#3498db' if v >= 0 else '#e74c3c' for v in nli_changes]
    
    bars1 = ax.bar(x - width/2, math_changes, width, label='Math Change',
                   color=math_colors, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, nli_changes, width, label='NLI Change',
                   color=nli_colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Change from Baseline (%)', fontsize=12)
    ax.set_title(f'Performance Change from Baseline (Math={baseline_math:.1f}%, NLI={baseline_nli:.1f}%)',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -12
        ax.annotate(f'{height:+.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, offset), textcoords="offset points", ha='center', va=va, fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -12
        ax.annotate(f'{height:+.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, offset), textcoords="offset points", ha='center', va=va, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_change.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'performance_change.png'}")


def create_summary_table(all_results: Dict[str, Dict], output_dir: Path):
    """Create a markdown summary table."""
    lines = [
        "# Experiment Results Summary\n",
        "| Experiment | Math % | NLI % | Math Acc | NLI Acc | Math Change | NLI Change | Training Time |",
        "|------------|--------|-------|----------|---------|-------------|------------|---------------|",
    ]
    
    baseline_math = 0
    baseline_nli = 0
    if "baseline" in all_results and "final_metrics" in all_results["baseline"]:
        baseline_math = all_results["baseline"]["final_metrics"].get("math_acc", 0)
        baseline_nli = all_results["baseline"]["final_metrics"].get("nli_acc", 0)
    
    order = ["baseline", "math-only", "nli-only", "mixed-1-1", "mixed-2-1", "mixed-3-1", "mixed-7-1", "mixed-15-1"]
    for exp_name in order:
        if exp_name not in all_results or "final_metrics" not in all_results[exp_name]:
            continue
        
        results = all_results[exp_name]
        
        # Calculate training percentages from config ratios
        if exp_name == "baseline":
            math_pct_str = "-"
            nli_pct_str = "-"
        elif exp_name == "math-only":
            math_pct_str = "100.0%"
            nli_pct_str = "0.0%"
        elif exp_name == "nli-only":
            math_pct_str = "0.0%"
            nli_pct_str = "100.0%"
        else:
            # Extract ratios from config
            config = results.get("config", {})
            math_ratio = config.get("math_ratio", 0)
            nli_ratio = config.get("nli_ratio", 0)
            total = math_ratio + nli_ratio
            if total > 0:
                math_pct = (math_ratio / total) * 100
                nli_pct = (nli_ratio / total) * 100
                math_pct_str = f"{math_pct:.1f}%"
                nli_pct_str = f"{nli_pct:.1f}%"
            else:
                math_pct_str = "-"
                nli_pct_str = "-"
        
        math_acc = results["final_metrics"].get("math_acc", 0) * 100
        nli_acc = results["final_metrics"].get("nli_acc", 0) * 100
        math_delta = (results["final_metrics"].get("math_acc", 0) - baseline_math) * 100
        nli_delta = (results["final_metrics"].get("nli_acc", 0) - baseline_nli) * 100
        
        train_time = results.get("training_time", 0)
        if train_time < 60:
            time_str = f"{train_time:.0f}s"
        elif train_time < 3600:
            time_str = f"{train_time/60:.1f}m"
        else:
            time_str = f"{train_time/3600:.1f}h"
        
        if exp_name == "baseline":
            math_delta_str = "-"
            nli_delta_str = "-"
        else:
            math_delta_str = f"+{math_delta:.1f}" if math_delta >= 0 else f"{math_delta:.1f}"
            nli_delta_str = f"+{nli_delta:.1f}" if nli_delta >= 0 else f"{nli_delta:.1f}"
        
        lines.append(
            f"| {exp_name} | {math_pct_str} | {nli_pct_str} | {math_acc:.1f}% | {nli_acc:.1f}% | "
            f"{math_delta_str} | {nli_delta_str} | {time_str} |"
        )
    
    summary = "\n".join(lines)
    
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)
    
    print(f"Saved: {output_dir / 'summary.md'}")
    print("\n" + summary)


def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    # Make default path relative to script location
    script_dir = Path(__file__).parent
    default_results_dir = script_dir / "outputs"
    
    parser.add_argument("--results-dir", type=str, default=str(default_results_dir),
                        help="Directory containing *_results.json files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (defaults to results-dir)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Plots from Experiment Results")
    print("=" * 60)
    print(f"Looking for results in: {results_dir.absolute()}")
    
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("\nNo results found! Run experiments first.")
        return
    
    print("\nGenerating plots...")
    plot_accuracy_comparison(all_results, output_dir)
    plot_pareto_frontier(all_results, output_dir)
    plot_performance_change(all_results, output_dir)
    create_summary_table(all_results, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()


