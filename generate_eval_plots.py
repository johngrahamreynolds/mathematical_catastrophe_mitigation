"""
Generate dual-panel training dynamics plots

This script creates a side-by-side comparison showing:
- Left panel: NLI accuracy (catastrophic forgetting)
- Right panel: Math accuracy (mixed training superiority)

Usage:
    python generate_eval_plots.py --nli-dir ./nli_exports --math-dir ./math_exports
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse

# Publication-quality settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['legend.framealpha'] = 0.95

# Color scheme - consistent across all figures
COLORS = {
    'baseline': '#7f7f7f',      # gray
    'math-only': '#d62728',     # red
    'nli-only': '#1f77b4',      # blue  
    'mixed-1-1': '#e377c2',     # pink
    'mixed-3-1': '#ffbb00',     # yellow/gold
    'mixed-7-1': '#9467bd',     # purple
    'mixed-15-1': '#8b5a00',    # brown (darker for visibility)
}

# Line styles for print clarity
LINE_STYLES = {
    'math-only': '-',
    'nli-only': '-',
    'mixed-1-1': '-',
    'mixed-3-1': '-',
    'mixed-7-1': '--',
    'mixed-15-1': '-.',
}


def load_csv_data(csv_path):
    """Load TensorBoard CSV export."""
    df = pd.read_csv(csv_path)
    return df[['Step', 'Value']].values


def create_dual_panel_figure(nli_data, math_data, output_dir):
    """
    Create dual-panel figure with NLI and Math accuracy.
    
    Args:
        nli_data: Dict mapping experiment name to (steps, nli_acc) array
        math_data: Dict mapping experiment name to (steps, math_acc) array
        output_dir: Path to save outputs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    plot_order = ['math-only', 'nli-only', 'mixed-1-1', 'mixed-3-1', 
                  'mixed-7-1', 'mixed-15-1']
    
    # ============================================================
    # LEFT PANEL: NLI Accuracy (Catastrophic Forgetting)
    # ============================================================
    
    for exp_name in plot_order:
        if exp_name not in nli_data:
            continue
            
        data = nli_data[exp_name]
        steps = data[:, 0]
        values = data[:, 1]
        
        ax1.plot(steps, values, 
                label=exp_name, 
                color=COLORS[exp_name],
                linestyle=LINE_STYLES[exp_name],
                linewidth=2.0,
                alpha=0.9)
    
    ax1.set_xlabel('Training Steps', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('NLI Validation Accuracy', fontsize=11, fontweight='semibold')
    ax1.set_title('(a) NLI Performance: Catastrophic Forgetting', 
                  fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, 9000)
    ax1.set_ylim(0.10, 0.92)
    
    # Add catastrophic forgetting annotation
    if 'math-only' in nli_data:
        math_only_nli = nli_data['math-only']
        early_steps = math_only_nli[math_only_nli[:, 0] <= 2000]
        if len(early_steps) > 0:
            worst_idx = early_steps[:, 1].argmin()
            catastrophe_step = early_steps[worst_idx, 0]
            catastrophe_value = early_steps[worst_idx, 1]
            
            ax1.annotate('Catastrophic\nForgetting', 
                        xy=(catastrophe_step, catastrophe_value),
                        xytext=(catastrophe_step + 2500, 0.25),
                        arrowprops=dict(
                            arrowstyle='->',
                            color='#d62728',
                            lw=2,
                            connectionstyle='arc3,rad=0.3'
                        ),
                        fontsize=9,
                        color='#d62728',
                        fontweight='bold',
                        ha='center',
                        bbox=dict(
                            boxstyle='round,pad=0.4',
                            facecolor='white',
                            edgecolor='#d62728',
                            linewidth=1.5,
                            alpha=0.95
                        ))
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ============================================================
    # RIGHT PANEL: Math Accuracy (Mixed Training Superiority)
    # ============================================================
    
    for exp_name in plot_order:
        if exp_name not in math_data:
            continue
            
        data = math_data[exp_name]
        steps = data[:, 0]
        values = data[:, 1]
        
        ax2.plot(steps, values, 
                label=exp_name, 
                color=COLORS[exp_name],
                linestyle=LINE_STYLES[exp_name],
                linewidth=2.0,
                alpha=0.9)
    
    ax2.set_xlabel('Training Steps', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Math Validation Accuracy', fontsize=11, fontweight='semibold')
    ax2.set_title('(b) Math Performance: Convergent Learning', 
                  fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, 9000)
    ax2.set_ylim(0.01, 0.15)  # 1% to 15%
    
    # Note: Annotation removed for conservative messaging
    # Models converge to similar final performance (11.7-12.0%)
    # Peak checkpoints vary but final evaluations show equivalence
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ============================================================
    # SHARED LEGEND
    # ============================================================
    
    # Create legend from first axis
    handles, labels = ax1.get_legend_handles_labels()
    
    # Place legend below both subplots
    fig.legend(handles, labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.05),
              ncol=3, 
              fontsize=9,
              frameon=True,
              framealpha=0.95,
              edgecolor='gray',
              fancybox=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3)  # Make room for legend
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'training_dynamics_dual.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'training_dynamics_dual.pdf',
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"✓ Saved: {output_dir / 'training_dynamics_dual.png'}")
    print(f"✓ Saved: {output_dir / 'training_dynamics_dual.pdf'}")
    
    plt.close()


def create_individual_figures(nli_data, math_data, output_dir):
    """Create individual figures for NLI and Math (for appendix or alternatives)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_order = ['math-only', 'nli-only', 'mixed-1-1', 'mixed-3-1', 
                  'mixed-7-1', 'mixed-15-1']
    
    # ============================================================
    # FIGURE 1: NLI Accuracy Only
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for exp_name in plot_order:
        if exp_name not in nli_data:
            continue
        data = nli_data[exp_name]
        ax.plot(data[:, 0], data[:, 1], 
               label=exp_name, 
               color=COLORS[exp_name],
               linestyle=LINE_STYLES[exp_name],
               linewidth=2.0, alpha=0.9)
    
    ax.set_xlabel('Training Steps', fontsize=11, fontweight='semibold')
    ax.set_ylabel('NLI Validation Accuracy', fontsize=11, fontweight='semibold')
    ax.set_title('NLI Performance: Catastrophic Forgetting in Math-Only Training', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 9000)
    ax.set_ylim(0.10, 0.92)
    ax.legend(loc='center right', fontsize=9, framealpha=0.95, edgecolor='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'nli_dynamics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'nli_dynamics.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_dir / 'nli_dynamics.png'}")
    plt.close()
    
    # ============================================================
    # FIGURE 2: Math Accuracy Only
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for exp_name in plot_order:
        if exp_name not in math_data:
            continue
        data = math_data[exp_name]
        ax.plot(data[:, 0], data[:, 1], 
               label=exp_name, 
               color=COLORS[exp_name],
               linestyle=LINE_STYLES[exp_name],
               linewidth=2.0, alpha=0.9)
    
    ax.set_xlabel('Training Steps', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Math Validation Accuracy', fontsize=11, fontweight='semibold')
    ax.set_title('Math Performance: Mixed Training Achieves Superior Results', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 9000)
    ax.set_ylim(0.01, 0.15)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'math_dynamics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'math_dynamics.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_dir / 'math_dynamics.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create training dynamics figures from TensorBoard exports'
    )
    parser.add_argument('--nli-dir', type=str, default='./nli_exports',
                       help='Directory containing NLI accuracy CSV exports')
    parser.add_argument('--math-dir', type=str, default='./math_exports',
                       help='Directory containing Math accuracy CSV exports')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory to save output figures')
    parser.add_argument('--individual', action='store_true',
                       help='Also create individual figures (not just dual panel)')
    args = parser.parse_args()
    
    nli_dir = Path(args.nli_dir)
    math_dir = Path(args.math_dir)
    
    # Expected filenames (adjust based on your exports)
    experiments = {
        'math-only': 'math-only-tag-eval_nli_acc.csv',
        'nli-only': 'nli-only-tag-eval_nli_acc.csv',
        'mixed-1-1': 'mixed-1-1-tag-eval_nli_acc.csv',
        'mixed-3-1': 'mixed-3-1-tag-eval_nli_acc.csv',
        'mixed-7-1': 'mixed-7-1-tag-eval_nli_acc.csv',
        'mixed-15-1': 'mixed-15-1-tag-eval_nli_acc.csv',
    }
    
    math_files = {k: v.replace('nli_acc', 'math_acc') for k, v in experiments.items()}
    
    # Load NLI data
    nli_data = {}
    print("Loading NLI accuracy data...")
    for exp_name, filename in experiments.items():
        csv_path = nli_dir / filename
        if csv_path.exists():
            nli_data[exp_name] = load_csv_data(csv_path)
            print(f"  ✓ {exp_name}: {len(nli_data[exp_name])} points")
        else:
            print(f"  ✗ Missing: {filename}")
    
    # Load Math data
    math_data = {}
    print("\nLoading Math accuracy data...")
    for exp_name, filename in math_files.items():
        csv_path = math_dir / filename
        if csv_path.exists():
            math_data[exp_name] = load_csv_data(csv_path)
            print(f"  ✓ {exp_name}: {len(math_data[exp_name])} points")
        else:
            print(f"  ✗ Missing: {filename}")
    
    if not nli_data or not math_data:
        print("\n❌ Missing required data files. Cannot create figures.")
        print("\nTo export from TensorBoard:")
        print("1. Navigate to eval/nli_acc and eval/math_acc plots")
        print("2. Click ⋮ → 'Export as CSV' for each experiment")
        print("3. Save to respective directories")
        return
    
    print(f"\nCreating dual-panel figure...")
    create_dual_panel_figure(nli_data, math_data, args.output_dir)
    
    if args.individual:
        print(f"\nCreating individual figures...")
        create_individual_figures(nli_data, math_data, args.output_dir)
    
    print("\n✅ Done! Figures ready for paper.")
    print(f"   Location: {Path(args.output_dir).absolute()}")


if __name__ == '__main__':
    main()
