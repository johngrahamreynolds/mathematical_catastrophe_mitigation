# Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training

[![Paper](https://img.shields.io/badge/arXiv-2512.13706-b31b1b)](https://arxiv.org/abs/2512.13706)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/collections/MarioBarbeque/catastrophic-forgetting-in-mathematical-reasoning)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Research code for:** *Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training*

**Author:** John Graham Reynolds  
**Institution:** The University of Texas at Austin

## 📋 Overview

This repository contains code for investigating **catastrophic forgetting** during finetuning of Flan-T5-Base for mathematical reasoning. We demonstrate that training exclusively on mathematical data dramatically improves math performance (3.1% → 12.0%) but causes severe forgetting of natural language understanding capabilities (81.0% → 16.5% on MultiNLI). 

We propose and evaluate **mixed training strategies** that interleave mathematical and NLI examples during training, successfully mitigating catastrophic forgetting while maintaining mathematical gains.

### Key Findings

- **Math-only training** achieves 12.0% math accuracy but drops NLI to 16.5% (64.5% absolute decrease)
- **Mixed training (1:1 ratio)** achieves equivalent math performance (12.0%) while maintaining 86.2% NLI accuracy
- Even minimal NLI exposure (6.2% in 15:1 ratio) provides sufficient regularization to prevent catastrophic forgetting
- All mixed training strategies achieve similar math performance, suggesting mixed training does not compromise specialized learning

## 🎯 Research Questions

1. How much does finetuning on math improve mathematical reasoning in Flan-T5-Base?
2. Does this finetuning cause forgetting of NLU capabilities?
3. Can mixed training (math + NLI) mitigate forgetting while maintaining math gains?

## 📊 Results Summary

All experiments conducted on Flan-T5-Base (250M parameters):

| Training Strategy | Math % | NLI % | Math Acc | NLI Acc | Math Δ | NLI Δ |
|-------------------|--------|-------|----------|---------|--------|-------|
| Baseline (pretrained) | — | — | 3.1% | 81.0% | — | — |
| **Math-only** | 100% | 0% | 12.0% | **16.5%** ⚠️ | +8.9 | **-64.5** |
| **NLI-only** | 0% | 100% | 1.6% | 86.9% | -1.5 | +5.9 |
| **Mixed 1:1** ⭐ | 50% | 50% | **12.0%** | **86.2%** | +8.9 | +5.2 |
| **Mixed 3:1** | 75% | 25% | 11.7% | 85.6% | +8.6 | +4.6 |
| **Mixed 7:1** | 87.5% | 12.5% | 11.7% | 84.5% | +8.6 | +3.5 |
| **Mixed 15:1** | 93.8% | 6.2% | 11.7% | 83.8% | +8.6 | +2.8 |

*Results evaluated on complete validation sets (Math: 10,000 examples; NLI: 9,815 examples)*

**Key Finding:** Mixed training (1:1 ratio) achieves **equivalent** math performance (12.0% vs 12.0%) while **completely eliminating** catastrophic forgetting (86.2% vs 16.5% NLI accuracy).

### Scaling Evidence

For comparison, our Flan-T5-Large (780M parameters) baseline achieves **90.8%** math accuracy—nearly **8× improvement** over Flan-T5-Base, suggesting significant capacity constraints at 250M parameters.

Model: [CyberSolve-LinAlg-1.2](https://huggingface.co/MarioBarbeque/CyberSolve-LinAlg-1.2)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/johngrahamreynolds/mathematical_catastrophe_mitigation.git
cd mathematical_catastrophe_mitigation

# Install dependencies
pip install -r requirements.txt
```

### Using Published Models

All trained models are available on HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load mixed training model (1:1 ratio)
model = T5ForConditionalGeneration.from_pretrained("MarioBarbeque/flan-t5-base-mixed-1-1-catastrophic")
tokenizer = T5Tokenizer.from_pretrained("MarioBarbeque/flan-t5-base-mixed-1-1-catastrophic")

# Math example
input_text = "Solve 24 = 1601*c - 1605*c for c."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # "-6"

# NLI example
input_text = "mnli premise: The cat sat on the mat. hypothesis: A cat was on a mat."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # "yes"
```

### Available Models

- `MarioBarbeque/flan-t5-base-math-only-catastrophic` - Math-only training (final checkpoint)
- `MarioBarbeque/flan-t5-base-nli-only-catastrophic` - NLI-only training (final checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-1-1-catastrophic` - Mixed training 1:1 (final checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-3-1-catastrophic` - Mixed training 3:1 (final checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-7-1-catastrophic` - Mixed training 7:1 (final checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-15-1-catastrophic` - Mixed training 15:1 (final checkpoint)

## 🔬 Reproducing Experiments

### Run Baseline Evaluation

```bash
python run_experiments.py --baseline-only
```

### Run Specific Experiment

```bash
# Math-only training
python run_experiments.py --experiment math-only

# Mixed training (1:1 ratio)
python run_experiments.py --experiment mixed-1-1

# Mixed training (3:1 ratio)
python run_experiments.py --experiment mixed-3-1
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=./logs

# Open http://localhost:6006 to view:
# - Training loss (total, math, NLI)
# - Learning rate schedule
# - Evaluation accuracy over time
```

### Generate Plots

After running experiments, export the tensorboard results to folders `/nli_exports`, `/math_exports` as csv files for each experiment type. The training quick-eval visualizations can then be generated with:

```bash
python generate_eval_plots.py --nli-dir ./nli_exports --math-dir ./math_exports
```

OR 

If you dont want to rerun the training, you can generate the plots with final research values available in `/tensorboard_exports` by running:

```bash
python generate_eval_plots.py --nli-dir ./tensorboard_exports --math-dir ./tensorboard_exports
```

The Pareto frontier plot is generated with:

```bash
python generate_acc_plots.py --results-dir ./outputs
```

As above, final research results are already populated in this repo if users are not interested in rerunning the trainings.

The `generate_eval_plots.py` script creates:
- `figures/training_dynamics.pdf` - NLI and Math performance checkpoints during training
- `figures/training_dynamics.png`

The `generate_acc_plots.py` script creates: 
- `outputs/accuracy_comparison.png` - Bar chart comparing all experiments
- `outputs/pareto_frontier.png` - Math vs NLI accuracy scatter plot
- `outputs/performance_change.png` - Delta from baseline
- `outputs/summary.md` - Markdown results table

## 📁 Project Structure

```
mathematical_catastrophe_mitigation/
├── config.py                  # Hyperparameters and experiment definitions
├── data.py                    # Dataset loading, preprocessing, mixed dataloaders
├── train.py                   # Training loop with logging and checkpointing
├── evaluate.py                # Evaluation (model.generate() + accuracy)
├── utils.py                   # Checkpointing, TensorBoard logging, utilities
├── run_experiments.py         # Main entry point for running experiments
├── colab.ipynb                # Colab nb with cell outputs of original training runs
├── generate_eval_plots.py     # Generate quick-eval plots of training dynamics
├── generate_acc_plots.py      # Generate accuracy plots from results
├── requirements.txt           # Dependencies
├── outputs/                   # Results, plots, and summaries (generated)
│   ├── all_results.json       # Aggregated experiment results
│   ├── accuracy_comparison.png
│   ├── pareto_frontier.png
│   ├── training_dynamics_dual.png
│   └── summary.md
└── paper/                     # LaTeX source for research paper and precompiled PDF
    ├── catastrophe_mitigation.pdf
    ├── catastrophe_mitigation.tex
    ├── references.bib
    └── compile.sh
```

## 📚 Datasets

### DeepMind Mathematics (Linear Algebra 1D)
- **Source:** `MarioBarbeque/DeepMind-LinAlg-1D-train` (pre-tokenized version of the 1D Lin Alg split)
- **Size:** ~392,702 training examples (subsampled to match NLI)
- **Format:** "Solve 24 = 1601*c - 1605*c for c." → "-6"

### MultiNLI
- **Source:** `multi_nli` from HuggingFace
- **Size:** 392,702 training examples, 9,815 validation examples
- **Format:** "mnli premise: {premise} hypothesis: {hypothesis}" → "yes"/"neutral"/"no"

## ⚙️ Configuration

Key hyperparameters (defined in `config.py`):

```python
# Model
model_name = "google/flan-t5-base"
dtype = torch.bfloat16

# Training
learning_rate = 3e-4
batch_size = 256  # Adjusted per experiment for consistent effective batch size
num_epochs = 3
gradient_accumulation_steps = 1

# Data
max_input_length = 128   # Covers 95th percentile of NLI
max_label_length = 8     # Covers all labels
```

## 📖 Paper

The full research paper is available in the `paper/` directory. To compile:

```bash
cd paper
./compile.sh
```

Or view the pre-compiled PDF: [`paper/catastrophe_mitigation.pdf`](paper/catastrophe_mitigation.pdf)

## 🤝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{reynolds2024mitigating,
  title={Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training},
  author={Reynolds, John Graham},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/johngrahamreynolds/mathematical_catastrophe_mitigation}
}
```

## 📝 License

This project is released under the Apache 2.0 License. See LICENSE file for details.

## 🙏 Acknowledgments

- Wonderful instruction from Greg Durrett and Philipp Krähenbühl
- Motivation from John Jumper's talk at Vanderbilt University
- Computational resources provided through Google Colab Pro
- HuggingFace for model hosting and dataset access

## 🔗 Links

- **Paper:** [arXiv/PDF](https://arxiv.org/abs/2412.XXXXX)
- **HuggingFace Models:** [MarioBarbeque](https://huggingface.co/collections/MarioBarbeque/catastrophic-forgetting-in-mathematical-reasoning)
- **Repository:** [GitHub](https://github.com/johngrahamreynolds/mathematical_catastrophe_mitigation)

---

For questions or issues, please open an issue on GitHub.

