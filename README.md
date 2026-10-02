# QTCL: Quantum-based Tensor Contraction Layers

**Hybrid quantum–classical prediction heads for compressed image classifiers**

Research implementation by **Van Tien Nguyen** and **Panagiotis (Panos) Markopoulos** for the SPIE *Machine Learning from Challenging Data (MLCD)* paper **“Quantum-based Tensor Contraction Layers.”**

[Manuscript](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14030/140300I/Quantum-based-tensor-contraction-layers/10.1117/12.3098024.short) · [Presentation slides](presentation/QTCL_MLCD2026_SPIE.pdf)

QTCL extends the classical Tensor Contraction Layer (TCL) by preserving its spatial contractions and compressing its channel map through two thin classical projections and a parameterized quantum circuit. The head blends classical features with quantum expectation values and trains jointly with a PyTorch image classifier.

- Quantum circuits use **PennyLane**, its `default.qubit` simulator, and `qml.qnn.TorchLayer` with the PyTorch interface. See the [PennyLane PyTorch documentation](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).
- Supported circuit families are `HEA`, `SLE`, `QAOA`, `MPS`, and `MERA`.
- Principal experiments compare fully connected, TCL, and QTCL heads on AlexNet and VGG19 with CIFAR-100. Additional drivers cover ResNet50, limited training data, and qubit/depth ablations.
- The supplied implementation runs quantum simulation locally. Head-parameter savings measure model compression; they do not establish a quantum runtime advantage.

## Installation

Use **Python 3.11** and run all commands from the repository root in a Bash-compatible shell.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) includes PyTorch, torchvision, PennyLane, Matplotlib, and tqdm. It specifies version ranges rather than an exact environment lock.

The quick start, circuit sanity check, and plotting commands were validated with Python 3.11, PyTorch `2.5.1+cpu`, torchvision `0.20.1+cpu`, and PennyLane `0.45.1`.

For a specific CPU/CUDA build, install a compatible PyTorch/torchvision pair using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/) before installing the requirements. Training selects CUDA when available and otherwise uses CPU.

Check the imports:

```bash
python -c "import torch, torchvision, pennylane, matplotlib, tqdm; print('Imports OK; PennyLane', pennylane.__version__)"
```

## Quick start

Run a short QTCL experiment with a small subset of CIFAR-100:

```bash
python -m script.exp.alexnet_cifar100 \
  --model alexnet_qtcl --epochs 1 --batch-size 16 \
  --train-fraction 0.01 --val-fraction 0.01 --num-workers 0 \
  --qtcl-n-qubits 4 --qtcl-n-layers 1 --qtcl-alpha 0.6 \
  --output-dir runs/quickstart
```

The first run downloads CIFAR-100 into `data/`. This small run checks the workflow; its accuracy is not intended to reproduce the manuscript. Quantum simulation can take longer than classical training.

| File | Contents |
| --- | --- |
| `training_log.csv` | Per-epoch training and validation loss/accuracy |
| `training_summary.csv` | Best and final metrics |
| `training_curves.png` | Loss and accuracy curves |
| `checkpoint_best_acc.pth` | Best validation-accuracy checkpoint, including model and training arguments |
| `run.log` | Console output |

Choose a new `--output-dir` to preserve a previous run; reusing a directory can overwrite its files. Use `python -m` for project entry points so imports resolve from the repository root.

## Experiments

### CIFAR-100 and model variants

CIFAR-100 downloads automatically under `--data-dir` (default `./data`). Training uses random cropping, horizontal flips, and normalization; evaluation uses normalization. `--train-fraction` and `--val-fraction` control the subsets.

**Evaluation convention:** the scripts label evaluation metrics as validation metrics, but evaluate on the official CIFAR-100 test split and use that split to select the best checkpoint.

| Variant | AlexNet | VGG19 |
| --- | --- | --- |
| Original classifier | `alexnet` | `vgg19` |
| First FC layer replaced by TCL | `alexnet_tcl` | `vgg19_tcl` |
| First two FC layers replaced by TCL | `alexnet_tcl12` | `vgg19_tcl12` |
| First FC layer replaced by QTCL | `alexnet_qtcl` | `vgg19_qtcl` |
| First two FC layers replaced by QTCL | `alexnet_qtcl12` | `vgg19_qtcl12` |

Single-layer replacements also accept `_tcl1` and `_qtcl1` suffixes. `--pretrained` initializes the backbone from ImageNet weights; by default models start from random initialization.

### AlexNet

```bash
python -m script.exp.alexnet_cifar100 \
  --model alexnet --output-dir runs/alexnet

python -m script.exp.alexnet_cifar100 \
  --model alexnet_tcl12 --output-dir runs/alexnet-tcl12

python -m script.exp.alexnet_cifar100 \
  --model alexnet_qtcl --qtcl-alpha 0.6 --output-dir runs/alexnet-qtcl1

python -m script.exp.alexnet_cifar100 \
  --model alexnet_qtcl12 --qtcl-alpha 0.6 --output-dir runs/alexnet-qtcl12
```

### VGG19

```bash
python -m script.exp.vgg19_cifar100 \
  --model vgg19 --num-seeds 1 --output-dir runs/vgg19

python -m script.exp.vgg19_cifar100 \
  --model vgg19_tcl12 --num-seeds 1 --output-dir runs/vgg19-tcl12

python -m script.exp.vgg19_cifar100 \
  --model vgg19_qtcl --qtcl-alpha 0.6 --num-seeds 1 \
  --output-dir runs/vgg19-qtcl1

python -m script.exp.vgg19_cifar100 \
  --model vgg19_qtcl12 --qtcl-alpha 0.5 --num-seeds 1 \
  --output-dir runs/vgg19-qtcl12
```

VGG19 defaults to **five consecutive seeds**, starting at `--seed 42`. Use `--num-seeds 1` for one run, or `--seeds 0 1 2 3 4` for explicit seeds. Multiple runs write per-seed artifacts under `seed_<seed>/`, plus `seed_summaries.csv`, `aggregate_summary.csv`, and `aggregate_training_log.csv`. The root directory also contains `experiment_config.json` and `run.log`.

These commands use current script defaults. Archived runs may use different settings; consult saved configurations, checkpoint arguments, and logs in [`results/`](results/) before attempting an exact reproduction.

### Training and quantum settings

AlexNet and VGG19 default to 160 epochs, training batches of 128, SGD with learning rate `0.01`, momentum `0.9`, and weight decay `1e-4`. Quantum parameters use a separate AdamW optimizer with learning rate `5e-3`. StepLR decays the learning rate by `0.1` every 30 epochs for AlexNet and every 40 epochs for VGG19.

| Option | Meaning / default |
| --- | --- |
| `--qtcl-n-qubits` | Circuit width, `8` |
| `--qtcl-n-layers` | Circuit depth, `2` |
| `--qtcl-F` | Measured quantum features, defaults to qubit count; use `1 <= F <= n_qubits` |
| `--qtcl-alpha` | Quantum mixing weight, `0.5`; `0` selects classical features and `1` selects quantum features |
| `--qtcl-learnable-alpha` | Train the mixing weight through a sigmoid parameterization |
| `--qtcl-shots` | `0` for analytic expectations; positive values enable finite-shot sampling |
| `--qtcl-ansatz` | `HEA` (default), `SLE`, `QAOA`, `MPS`, or `MERA` |
| `--qtcl-ansatz-kwargs` | JSON object passed to the circuit builder |
| `--qtcl-lr` | Quantum-parameter learning rate |
| `--freeze-backbone-epochs` | Initial epochs with backbone SGD updates frozen |

For two QTCL blocks, second-block overrides include `--qtcl2-n-qubits`, `--qtcl2-n-layers`, `--qtcl2-F`, `--qtcl2-alpha`, `--qtcl2-learnable-alpha`, `--qtcl2-shots`, `--qtcl2-ansatz`, and `--qtcl2-ansatz-kwargs`. See each driver's `--help` for inheritance rules and all available options.

### Qubit/depth and limited-data studies

```bash
python -m script.exp.ablation_study_qubit_depth \
  --qubits 4 8 --depths 1 2 --epochs 160 --output-dir runs/qubit-depth

python -m script.exp.limited_training_data \
  --methods vgg19 tcl12 qtcl12 --fractions 0.1 0.25 0.5 1.0 \
  --epochs 160 --output-dir runs/limited-data
```

These sweeps launch multiple VGG19 training runs and can take substantially longer than the quick start. ResNet50 has separate training and checkpoint-evaluation drivers under `script/exp/` and `script/eval/`; inspect their `--help` for its model variants.

### Multiple GPUs

Training drivers detect distributed execution through `RANK` and `WORLD_SIZE`. For two CUDA GPUs:

```bash
torchrun --standalone --nproc_per_node=2 --module script.exp.alexnet_cifar100 \
  --model alexnet_qtcl12 --qtcl-alpha 0.6 --output-dir runs/alexnet-qtcl12-ddp
```

## Evaluation and plotting

Evaluate an original-classifier checkpoint produced by the AlexNet baseline run:

```bash
python -m script.eval.alexnet_cifar100 \
  --checkpoint runs/alexnet/checkpoint_best_acc.pth --num-workers 0
```

Equivalent evaluation drivers exist for VGG19 and ResNet50. They reconstruct the model from saved training arguments; `--val-fraction 1.0` requests the full evaluation split.

**Known limitation:** evaluation of the quick-start QTCL checkpoint currently fails with unexpected `U1`/`U2` state-dictionary keys because the evaluator loads weights before initializing the layer's spatial parameters. Training and its recorded validation metrics work; consult `training_log.csv` and `training_summary.csv` for those results.

Training automatically plots each run's curves. To regenerate the manuscript's accuracy/compression trade-off figures:

```bash
python -m script.plot.plot_alexnet_cifar100 --output-dir runs/figures
python -m script.plot.plot_vgg19_cifar100 --output-dir runs/figures
```

These scripts read the fixed tables [`alex.tex`](script/plot/alex.tex) and [`vgg19.tex`](script/plot/vgg19.tex), rather than recomputing values from new training runs. Each writes PNG and PDF figures; use `--tex-path` to supply another table.

## Results reported in the manuscript

These VGG19 values are reported in the manuscript linked above and included in the [plotting tables](script/plot/vgg19.tex). Savings refer to **head parameters**, relative to the uncompressed classifier.

| VGG19 head | Top-1 accuracy (%) | Head-parameter reduction (%) |
| --- | ---: | ---: |
| Original FC classifier | 72.21 | — |
| TCL, first two FC layers replaced | 67.48 | 97.27 |
| QTCL, first two FC layers replaced | 71.67 | 98.68 |

At aggressive compression, QTCL improves on TCL by 4.19 percentage points. Archived metrics, training curves, and sweep outputs are in [`results/`](results/). These reported values are not recomputed by the quick start and are not multi-seed means with uncertainty estimates.

## Repository layout

```text
src/
  tcl/tcl.py                     Classical tensor contraction layer
  qtcl/qtcl.py                   PennyLane/PyTorch quantum tensor contraction layer
  qtcl/ansatz.py                 HEA, SLE, QAOA, MPS, and MERA circuit builders
script/
  exp/                          Training and sweep entry points
  eval/                         Checkpoint evaluation drivers
  plot/                         Trade-off plotting scripts and LaTeX source tables
data/                           Local CIFAR-100 downloads and data artifacts
results/                        Archived experiment metrics and figures
presentation/                   MLCD presentation slides
requirements.txt                Training, simulation, and plotting dependencies
```

## Validation

Check the registered PennyLane circuits and inspect the training options:

```bash
python -m src.qtcl.ansatz
python -m script.exp.alexnet_cifar100 --help
python -m script.exp.vgg19_cifar100 --help
```

The ansatz module is a standalone circuit sanity check. The repository currently has no automated unit-test suite; use quick-start training to check the QTCL workflow. The checkpoint-evaluation limitation is described above.

## Citation

```bibtex
@inproceedings{10.1117/12.3098024,
author = {Van Tien Nguyen and Panagiotis (Panos) Markopoulos},
title = {{Quantum-based tensor contraction layers}},
volume = {14030},
booktitle = {Machine Learning from Challenging Data 2026},
editor = {Panagiotis  (Panos) Markopoulos and George Sklivanitis and Bing Ouyang},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {140300I},
keywords = {quantum , machine learning, tensors},
year = {2026},
doi = {10.1117/12.3098024},
URL = {https://doi.org/10.1117/12.3098024}
}
```

## Contact

**Van Tien Nguyen**

[tien.nguyen@utsa.edu](mailto:tien.nguyen@utsa.edu) · [vantn.prof@gmail.com](mailto:vantn.prof@gmail.com)

[Personal website](https://vantnprof.github.io)
