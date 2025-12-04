# Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://github.com/johngrahamreynolds/mathematical_catastrophe_mitigation)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/MarioBarbeque)

**Research code for:** *Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training*

**Author:** John Graham Reynolds  
**Institution:** University of Texas at Austin

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

| Experiment | Math % | NLI % | Math Acc | NLI Acc | Math Δ | NLI Δ |
|------------|--------|-------|----------|---------|--------|-------|
| **baseline** | - | - | 3.1% | 81.0% | - | - |
| **math-only** | 100.0% | 0.0% | 12.0% | 16.5% | +8.9 | -64.5 |
| **nli-only** | 0.0% | 100.0% | 1.6% | 86.9% | -1.6 | +5.9 |
| **mixed-1-1** | 50.0% | 50.0% | 12.0% | 86.2% | +8.9 | +5.3 |
| **mixed-3-1** | 75.0% | 25.0% | 11.7% | 85.6% | +8.6 | +4.6 |
| **mixed-7-1** | 87.5% | 12.5% | 11.7% | 84.5% | +8.6 | +3.5 |
| **mixed-15-1** | 93.8% | 6.2% | 11.7% | 83.8% | +8.6 | +2.8 |

*Results evaluated on complete validation sets (Math: 10,000 examples; NLI: 9,815 examples)*

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

- `MarioBarbeque/flan-t5-base-math-only-catastrophic` - Math-only training (best checkpoint)
- `MarioBarbeque/flan-t5-base-nli-only-catastrophic` - NLI-only training (best checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-1-1-catastrophic` - Mixed training 1:1 (best checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-3-1-catastrophic` - Mixed training 3:1 (best checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-7-1-catastrophic` - Mixed training 7:1 (best checkpoint)
- `MarioBarbeque/flan-t5-base-mixed-15-1-catastrophic` - Mixed training 15:1 (best checkpoint)

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

### Generate Plots

After running experiments, generate publication-ready visualizations:

```bash
python generate_plots.py --results-dir ./outputs
```

This creates:
- `accuracy_comparison.png` - Bar chart comparing all experiments
- `pareto_frontier.png` - Math vs NLI accuracy scatter plot
- `performance_change.png` - Delta from baseline
- `summary.md` - Markdown results table

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=./logs

# Open http://localhost:6006 to view:
# - Training loss (total, math, NLI)
# - Learning rate schedule
# - Evaluation accuracy over time
```

## 📁 Project Structure

```
mathematical_catastrophe_mitigation/
├── config.py              # Hyperparameters and experiment definitions
├── data.py                # Dataset loading, preprocessing, mixed dataloaders
├── train.py               # Training loop with logging and checkpointing
├── evaluate.py            # Evaluation (model.generate() + accuracy)
├── utils.py               # Checkpointing, TensorBoard logging, utilities
├── run_experiments.py     # Main entry point for running experiments
├── generate_plots.py      # Generate publication-ready plots from results
├── publish_to_hub.py      # Push trained models to HuggingFace Hub
├── requirements.txt       # Dependencies
├── outputs/               # Results, plots, and summaries (generated)
│   ├── all_results.json   # Aggregated experiment results
│   ├── accuracy_comparison.png
│   ├── pareto_frontier.png
│   └── summary.md
└── paper/                 # LaTeX source for research paper
    ├── catastrophe_mitigation.tex
    ├── references.bib
    └── compile.sh
```

## 📚 Datasets

### DeepMind Mathematics (Linear Algebra 1D)
- **Source:** `MarioBarbeque/DeepMind-LinAlg-1D-train` (pre-tokenized)
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
gradient_accumulation_steps = 1  # Adjusted per experiment

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
  year={2024},
  url={https://github.com/johngrahamreynolds/mathematical_catastrophe_mitigation}
}
```

## 📝 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Wonderful instruction from Greg Durrett
- Motivation from John Jumper's talk at Vanderbilt University
- Computational resources provided through Google Colab Pro
- HuggingFace for model hosting and dataset access

## 🔗 Links

- **Paper:** [arXiv/PDF](paper/catastrophe_mitigation.pdf)
- **HuggingFace Models:** [MarioBarbeque](https://huggingface.co/MarioBarbeque)
- **Repository:** [GitHub](https://github.com/johngrahamreynolds/mathematical_catastrophe_mitigation)

---

For questions or issues, please open an issue on GitHub.

