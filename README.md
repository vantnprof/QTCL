# QTCL: Quantum-based Tensor Contraction Layers

PyTorch implementation of tensor-contraction classifiers and hybrid quantum-classical tensor-contraction classifiers for CIFAR-100.

This repository includes:
- classical Tensor Contraction Layers (TCL),
- Quantum TCL (QTCL) with configurable quantum ansatz and mixing coefficient,
- experiment pipelines for AlexNet and VGG19 with reproducible CSV/plot outputs.

## Overview

The experiment entry points are:
- `script/exp/alexnet_cifar100.py`
- `script/exp/vgg19_cifar100.py`

Each script:
- loads CIFAR-100 with standard augmentation,
- swaps one or two fully connected classifier layers with TCL/QTCL blocks,
- trains with SGD (and AdamW for quantum parameters when QTCL is enabled),
- writes metrics, curves, and best checkpoints to `results/<experiment_name>/`.

## Repository Structure

```text
QTCL/
├── script/exp/
│   ├── alexnet_cifar100.py       # AlexNet experiments
│   └── vgg19_cifar100.py         # VGG19 experiments
├── src/tcl/tcl.py                # Classical tensor contraction layer
├── src/qtcl/qtcl.py              # Quantum TCL layer
├── src/qtcl/ansatz.py            # Registered ansatz builders (HEA/SLE/QAOA/MPS/MERA)
└── results/                      # Experiment outputs (CSV, checkpoint, curves)
```

## Model Variants

Both experiment scripts support the same variant pattern:

| Variant suffix | Meaning |
|---|---|
| `(base)` | Original classifier (no TCL/QTCL replacement) |
| `_tcl` or `_tcl1` | Replace first FC layer with TCL |
| `_tcl12` | Replace first and second FC layers with TCL |
| `_qtcl` or `_qtcl1` | Replace first FC layer with QTCL |
| `_qtcl12` | Replace first and second FC layers with QTCL |

Examples:
- AlexNet: `alexnet`, `alexnet_tcl`, `alexnet_tcl12`, `alexnet_qtcl`, `alexnet_qtcl12`
- VGG19: `vgg19`, `vgg19_tcl`, `vgg19_tcl12`, `vgg19_qtcl`, `vgg19_qtcl12`

## Environment Setup

1. Create and activate a virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install PyTorch (select the wheel matching your CUDA/CPU setup from PyTorch official docs), then install project dependencies:
```bash
pip install torch torchvision
pip install pennylane matplotlib tqdm
```

3. Optional sanity check:
```bash
python -c "import torch, torchvision, pennylane, matplotlib, tqdm; print('ok')"
```

## Data and Preprocessing

- Dataset: CIFAR-100 (auto-downloaded under `--data-dir`, default `./data`)
- Train transform: random crop (32, padding=4) + random horizontal flip + normalization
- Validation transform: normalization only
- Subsampling controls:
  - `--train-fraction` (default `1.0`)
  - `--val-fraction` (default `1.0`)

## Reproducible Commands

All commands below are from repository root.

### AlexNet Runs

```bash
# Baseline
python script/exp/alexnet_cifar100.py --model alexnet --output-dir results/alexnet

# TCL
python script/exp/alexnet_cifar100.py --model alexnet_tcl --output-dir results/alexnet_tcl
python script/exp/alexnet_cifar100.py --model alexnet_tcl12 --output-dir results/alexnet_tcl12

# QTCL (single replacement)
python script/exp/alexnet_cifar100.py --model alexnet_qtcl --qtcl-alpha 0.6 --output-dir results/alexnet_qtcl_alpha0.6
python script/exp/alexnet_cifar100.py --model alexnet_qtcl --qtcl-alpha 0.7 --output-dir results/alexnet_qtcl_alpha0.7

# QTCL (two replacements)
python script/exp/alexnet_cifar100.py --model alexnet_qtcl12 --qtcl-alpha 0.6 --output-dir results/alexnet_qtcl12_alpha0.6
python script/exp/alexnet_cifar100.py --model alexnet_qtcl12 --qtcl-alpha 0.9 --output-dir results/alexnet_qtcl12_alpha0.9
```

### VGG19 Runs

```bash
# Baseline (not yet present in current results folder, but supported)
python script/exp/vgg19_cifar100.py --model vgg19 --output-dir results/vgg19

# TCL
python script/exp/vgg19_cifar100.py --model vgg19_tcl --output-dir results/vgg19_tcl
python script/exp/vgg19_cifar100.py --model vgg19_tcl12 --output-dir results/vgg19_tcl12

# QTCL (single replacement)
python script/exp/vgg19_cifar100.py --model vgg19_qtcl --qtcl-alpha 0.5 --output-dir results/vgg19_qtcl_alpha0.5
python script/exp/vgg19_cifar100.py --model vgg19_qtcl --qtcl-alpha 0.6 --output-dir results/vgg19_qtcl_alpha0.6

# QTCL (two replacements)
python script/exp/vgg19_cifar100.py --model vgg19_qtcl12 --qtcl-alpha 0.4 --output-dir results/vgg19_qtcl12_alpha0.4
python script/exp/vgg19_cifar100.py --model vgg19_qtcl12 --qtcl-alpha 0.5 --output-dir results/vgg19_qtcl12_alpha0.5
```

### Multi-GPU (DDP via torchrun)

Scripts automatically enable distributed mode when `RANK` and `WORLD_SIZE` are set by `torchrun`.

```bash
torchrun --standalone --nproc_per_node=2 script/exp/alexnet_cifar100.py \
  --model alexnet_qtcl12 --qtcl-alpha 0.6 \
  --output-dir results/alexnet_qtcl12_alpha0.6
```

## Key Training Settings

| Setting | AlexNet script | VGG19 script |
|---|---|---|
| Epochs | 160 | 160 |
| Batch size | 128 train / 256 val | 128 train / 256 val |
| Backbone optimizer | SGD (lr=0.01, momentum=0.9, weight_decay=1e-4) | SGD (lr=0.01, momentum=0.9, weight_decay=1e-4) |
| Quantum optimizer (if QTCL exists) | AdamW (lr=5e-3, betas=(0.9,0.999), wd=1e-4) | AdamW (lr=5e-3, betas=(0.9,0.999), wd=1e-4) |
| LR scheduler | StepLR(step_size=30, gamma=0.1) | StepLR(step_size=40, gamma=0.1) |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

## QTCL Configuration

Main QTCL controls:
- `--qtcl-n-qubits` (default: `8`)
- `--qtcl-n-layers` (default: `2`)
- `--qtcl-F` (latent quantum feature width, default: `n_qubits`)
- `--qtcl-alpha` in `[0,1]` (classical/quantum interpolation weight)
- `--qtcl-learnable-alpha` (make alpha trainable)
- `--qtcl-shots` (`0` means analytic expectation)
- `--qtcl-ansatz` in `{HEA, SLE, QAOA, MPS, MERA}`
- `--qtcl-ansatz-kwargs` as JSON string
- `--freeze-backbone-epochs` to warm up QTCL while freezing backbone SGD updates

Second QTCL stage (`*_qtcl12`) overrides:
- `--qtcl2-n-qubits`, `--qtcl2-n-layers`, `--qtcl2-F`
- `--qtcl2-alpha`, `--qtcl2-learnable-alpha`
- `--qtcl2-shots`, `--qtcl2-ansatz`, `--qtcl2-ansatz-kwargs`

Note:
- `--qtcl-batchnorm` is used in the two-stage QTCL path (`*_qtcl12`).
- Single-stage QTCL replacement (`*_qtcl1`) currently injects BatchNorm in script logic.

## Output Artifacts

Each run writes to `--output-dir`:
- `training_log.csv`: per-epoch train/val loss and accuracy
- `training_summary.csv`: best/final metrics summary
- `training_curves.png`: loss and accuracy curves
- `checkpoint_best_acc.pth`: best validation-accuracy checkpoint

For clean logs suitable for appendices:
```bash
python script/exp/alexnet_cifar100.py ... --no-progress 2>&1 | tee results/<exp_name>/log.txt
```

## Current Results Snapshot

From `results/*/training_summary.csv`:

Note: these are repository snapshot runs and may use different hardware/process counts; use the command sections above for controlled re-runs.

| Experiment | Best Val Acc (%) | Best Epoch | Best Val Loss |
|---|---:|---:|---:|
| alexnet | 65.70 | 79 | 1.3855 |
| alexnet_qtcl12_alpha0.6 | 57.92 | 80 | 1.7177 |
| alexnet_qtcl12_alpha0.9 | 57.89 | 62 | 1.7975 |
| alexnet_qtcl_alpha0.6 | 63.53 | 67 | 1.4396 |
| alexnet_qtcl_alpha0.7 | 63.39 | 73 | 1.4440 |
| alexnet_tcl | 66.56 | 88 | 1.2951 |
| alexnet_tcl12 | 65.69 | 65 | 1.3334 |
| vgg19_qtcl12_alpha0.4 | 71.02 | 152 | 1.4282 |
| vgg19_qtcl12_alpha0.5 | 71.67 | 99 | 1.3850 |
| vgg19_qtcl_alpha0.5 | 71.28 | 139 | 1.3812 |
| vgg19_qtcl_alpha0.6 | 71.47 | 130 | 1.3633 |
| vgg19_tcl | 71.83 | 53 | 1.3042 |
| vgg19_tcl12 | 67.48 | 158 | 1.4580 |

## Publication-Friendly Reporting Checklist

When preparing paper tables/appendix:
- record the exact command line and git commit hash,
- report `best_val_acc`, `best_val_acc_epoch`, and final metrics from `training_summary.csv`,
- include `training_curves.png` and selected `training_log.csv` slices for convergence evidence,
- if using DDP, report number of GPUs and `--dist-backend`,
- if comparing QTCL settings, report ansatz, qubits, layers, shots, and alpha policy (fixed vs learnable).

## Citation Template

If you use this codebase in a manuscript, cite your corresponding paper and this repository. Replace placeholders below:

```bibtex
@misc{qtcl_repo,
  title        = {QTCL: Quantum-based Tensor Contraction Layers},
  author       = {<Author List>},
  year         = {<Year>},
  howpublished = {\url{<Repository URL>}},
  note         = {Code for AlexNet/VGG19 CIFAR-100 TCL and QTCL experiments}
}
```
