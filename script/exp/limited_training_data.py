from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_MODULE = "script.exp.vgg19_cifar100"
FULL_VAL_FRACTION = 1.0
DEFAULT_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
DEFAULT_METHODS = ("vgg19", "tcl12", "qtcl12")
EXPECTED_QTCL12_SPACE_SAVING = 0.9868
EXPECTED_QTCL12_SPACE_SAVING_TOL = 5e-4


@dataclass(frozen=True)
class MethodSpec:
    key: str
    model: str
    qtcl_blocks: int = 0


METHOD_SPECS = {
    "vgg19": MethodSpec(key="vgg19", model="vgg19", qtcl_blocks=0),
    "tcl12": MethodSpec(key="tcl12", model="vgg19_tcl12", qtcl_blocks=0),
    "qtcl12": MethodSpec(key="qtcl12", model="vgg19_qtcl12", qtcl_blocks=2),
}


SUMMARY_FIELD_ORDER = [
    "method",
    "model",
    "train_fraction",
    "val_fraction",
    "seed",
    "qtcl_alpha",
    "qtcl2_alpha",
    "initialization",
    "run_output_dir",
    "best_val_acc",
    "best_val_acc_epoch",
    "best_val_loss",
    "best_val_loss_epoch",
    "final_train_loss",
    "final_train_acc",
    "final_val_loss",
    "final_val_acc",
]


def resolve_repo_path(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else (REPO_ROOT / path_like)


def fraction_label(fraction: float) -> str:
    return f"{fraction:.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the limited-training-data CIFAR100 fine-tuning experiment for VGG19, TCL, and QTCL variants. "
            "Each method starts either from ImageNet pretrained VGG19 weights or from random initialization, "
            "then trains on a training subset while validation always uses the full CIFAR100 validation split."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Dataset root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/limited_training_data"),
        help="Root directory for all per-run artifacts and sweep summaries.",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="Training-set fractions to evaluate.",
    )
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=list(DEFAULT_METHODS),
        default=None,
        help="Method variants to evaluate.",
    )
    method_group.add_argument(
        "--method",
        type=str,
        choices=list(DEFAULT_METHODS),
        default=None,
        help="Run exactly one method variant.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Single seed to use for every run.")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate for the main optimizer.")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer used for the trainable parameters during fine-tuning.",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for SGD.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained VGG19 weights. When omitted, training starts from random initialization.",
    )
    parser.add_argument(
        "--tcl1-out-channels",
        type=int,
        default=512,
        help="Output channels for the first TCL/QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl1-out-height",
        type=int,
        default=3,
        help="Output height for the first TCL/QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl1-out-width",
        type=int,
        default=3,
        help="Output width for the first TCL/QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl2-out-channels",
        type=int,
        default=512,
        help="Output channels for the second TCL/QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl2-out-height",
        type=int,
        default=3,
        help="Output height for the second TCL/QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl2-out-width",
        type=int,
        default=3,
        help="Output width for the second TCL/QTCL replacement block.",
    )
    parser.add_argument("--qtcl-n-qubits", type=int, default=8, help="Number of qubits for QTCL blocks.")
    parser.add_argument("--qtcl-n-layers", type=int, default=2, help="Number of layers for QTCL blocks.")
    parser.add_argument(
        "--qtcl-F",
        type=int,
        default=None,
        help="Optional QTCL latent feature width F for the first QTCL block.",
    )
    parser.add_argument("--qtcl-alpha", type=float, default=0.5, help="Fixed alpha for QTCL runs.")
    parser.add_argument(
        "--qtcl-shots",
        type=int,
        default=0,
        help="Number of quantum shots for the first QTCL block (0 means analytic expectation).",
    )
    parser.add_argument("--qtcl-ansatz", type=str, default="HEA", help="Ansatz type for the first QTCL block.")
    parser.add_argument(
        "--qtcl-ansatz-kwargs",
        type=str,
        default="{}",
        help="JSON object of extra kwargs for the first QTCL ansatz.",
    )
    parser.add_argument(
        "--qtcl2-n-qubits",
        type=int,
        default=None,
        help="Optional qubit override for the second QTCL block.",
    )
    parser.add_argument(
        "--qtcl2-n-layers",
        type=int,
        default=None,
        help="Optional layer override for the second QTCL block.",
    )
    parser.add_argument(
        "--qtcl2-F",
        type=int,
        default=None,
        help="Optional latent feature width override for the second QTCL block.",
    )
    parser.add_argument(
        "--qtcl2-shots",
        type=int,
        default=None,
        help="Optional shots override for the second QTCL block.",
    )
    parser.add_argument(
        "--qtcl2-ansatz",
        type=str,
        default=None,
        help="Optional ansatz override for the second QTCL block.",
    )
    parser.add_argument(
        "--qtcl2-ansatz-kwargs",
        type=str,
        default=None,
        help="Optional JSON kwargs override for the second QTCL block.",
    )
    parser.add_argument(
        "--qtcl-lr",
        type=float,
        default=5e-3,
        help="Learning rate for the dedicated QTCL optimizer.",
    )
    parser.add_argument(
        "--qtcl-optimizer",
        type=str,
        choices=["adamw", "adam"],
        default="adamw",
        help="Optimizer used for trainable QTCL parameters during fine-tuning.",
    )
    parser.add_argument(
        "--qtcl-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the dedicated QTCL optimizer.",
    )
    parser.add_argument(
        "--qtcl-betas",
        type=float,
        nargs=2,
        metavar=("BETA1", "BETA2"),
        default=(0.9, 0.999),
        help="AdamW beta values for the dedicated QTCL optimizer.",
    )
    parser.add_argument(
        "--qtcl-batchnorm",
        dest="qtcl_batchnorm",
        action="store_true",
        help="Retain BatchNorm around the two-stage QTCL head.",
    )
    parser.add_argument(
        "--no-qtcl-batchnorm",
        dest="qtcl_batchnorm",
        action="store_false",
        help="Disable BatchNorm around the two-stage QTCL head.",
    )
    parser.set_defaults(qtcl_batchnorm=True)
    parser.add_argument("--lr-step-size", type=int, default=40, help="Epoch interval for StepLR.")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Multiplicative decay for StepLR.")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Initial epochs where the backbone SGD optimizer is frozen.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force retraining even if a per-run training summary already exists.",
    )
    args = parser.parse_args()
    args.qtcl_betas = tuple(args.qtcl_betas)
    args.data_dir = resolve_repo_path(Path(args.data_dir))
    args.output_dir = resolve_repo_path(Path(args.output_dir))
    if args.method is not None:
        args.methods = [args.method]
    elif args.methods is None:
        args.methods = list(DEFAULT_METHODS)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative.")
    if args.lr_step_size <= 0:
        raise ValueError("lr-step-size must be a positive integer.")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("lr-gamma must lie in the interval (0, 1].")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze-backbone-epochs must be non-negative.")
    if args.qtcl_n_qubits <= 0:
        raise ValueError("qtcl-n-qubits must be a positive integer.")
    if args.qtcl_n_layers <= 0:
        raise ValueError("qtcl-n-layers must be a positive integer.")
    if args.qtcl_shots < 0:
        raise ValueError("qtcl-shots must be non-negative.")
    if not 0.0 <= args.qtcl_alpha <= 1.0:
        raise ValueError("qtcl-alpha must lie in the interval [0, 1].")
    if args.qtcl_F is not None and args.qtcl_F <= 0:
        raise ValueError("qtcl-F must be a positive integer when provided.")
    if args.qtcl2_n_qubits is not None and args.qtcl2_n_qubits <= 0:
        raise ValueError("qtcl2-n-qubits must be a positive integer when provided.")
    if args.qtcl2_n_layers is not None and args.qtcl2_n_layers <= 0:
        raise ValueError("qtcl2-n-layers must be a positive integer when provided.")
    if args.qtcl2_F is not None and args.qtcl2_F <= 0:
        raise ValueError("qtcl2-F must be a positive integer when provided.")
    if args.qtcl2_shots is not None and args.qtcl2_shots < 0:
        raise ValueError("qtcl2-shots must be non-negative when provided.")
    if args.tcl1_out_channels <= 0 or args.tcl1_out_height <= 0 or args.tcl1_out_width <= 0:
        raise ValueError("First replacement dimensions must all be positive.")
    if args.tcl2_out_channels <= 0 or args.tcl2_out_height <= 0 or args.tcl2_out_width <= 0:
        raise ValueError("Second replacement dimensions must all be positive.")
    if len(args.fractions) == 0:
        raise ValueError("At least one training fraction must be provided.")
    for fraction in args.fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Every training fraction must lie in the interval (0, 1].")
    try:
        first_ansatz_kwargs = json.loads(args.qtcl_ansatz_kwargs)
    except json.JSONDecodeError as exc:
        raise ValueError("qtcl-ansatz-kwargs must decode as a JSON object.") from exc
    if not isinstance(first_ansatz_kwargs, dict):
        raise ValueError("qtcl-ansatz-kwargs must decode as a JSON object.")
    if args.qtcl2_ansatz_kwargs is not None:
        try:
            second_ansatz_kwargs = json.loads(args.qtcl2_ansatz_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError("qtcl2-ansatz-kwargs must decode as a JSON object.") from exc
        if not isinstance(second_ansatz_kwargs, dict):
            raise ValueError("qtcl2-ansatz-kwargs must decode as a JSON object.")

def build_run_output_dir(output_dir: Path, method: str, fraction: float) -> Path:
    return output_dir / method / f"fraction_{fraction_label(fraction)}"


def child_command(args: argparse.Namespace, spec: MethodSpec, fraction: float, run_output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        TRAINING_MODULE,
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(run_output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--optimizer",
        str(args.optimizer),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--model",
        spec.model,
        "--tcl1-out-channels",
        str(args.tcl1_out_channels),
        "--tcl1-out-height",
        str(args.tcl1_out_height),
        "--tcl1-out-width",
        str(args.tcl1_out_width),
        "--tcl2-out-channels",
        str(args.tcl2_out_channels),
        "--tcl2-out-height",
        str(args.tcl2_out_height),
        "--tcl2-out-width",
        str(args.tcl2_out_width),
        "--qtcl-n-qubits",
        str(args.qtcl_n_qubits),
        "--qtcl-n-layers",
        str(args.qtcl_n_layers),
        "--qtcl-alpha",
        str(args.qtcl_alpha),
        "--qtcl-shots",
        str(args.qtcl_shots),
        "--qtcl-ansatz",
        args.qtcl_ansatz,
        "--qtcl-ansatz-kwargs",
        args.qtcl_ansatz_kwargs,
        "--qtcl-lr",
        str(args.qtcl_lr),
        "--qtcl-optimizer",
        str(args.qtcl_optimizer),
        "--qtcl-weight-decay",
        str(args.qtcl_weight_decay),
        "--qtcl-betas",
        str(args.qtcl_betas[0]),
        str(args.qtcl_betas[1]),
        "--train-fraction",
        fraction_label(fraction),
        "--val-fraction",
        fraction_label(FULL_VAL_FRACTION),
        "--seeds",
        str(args.seed),
        "--lr-step-size",
        str(args.lr_step_size),
        "--lr-gamma",
        str(args.lr_gamma),
        "--freeze-backbone-epochs",
        str(args.freeze_backbone_epochs),
    ]

    if args.pretrained:
        command.append("--pretrained")
    if args.no_progress:
        command.append("--no-progress")
    if args.qtcl_batchnorm:
        command.append("--qtcl-batchnorm")
    if args.qtcl_F is not None:
        command.extend(["--qtcl-F", str(args.qtcl_F)])
    if args.qtcl2_n_qubits is not None:
        command.extend(["--qtcl2-n-qubits", str(args.qtcl2_n_qubits)])
    if args.qtcl2_n_layers is not None:
        command.extend(["--qtcl2-n-layers", str(args.qtcl2_n_layers)])
    if args.qtcl2_F is not None:
        command.extend(["--qtcl2-F", str(args.qtcl2_F)])
    if args.qtcl2_shots is not None:
        command.extend(["--qtcl2-shots", str(args.qtcl2_shots)])
    if args.qtcl2_ansatz is not None:
        command.extend(["--qtcl2-ansatz", args.qtcl2_ansatz])
    if args.qtcl2_ansatz_kwargs is not None:
        command.extend(["--qtcl2-ansatz-kwargs", args.qtcl2_ansatz_kwargs])

    if spec.qtcl_blocks > 0:
        command.append("--qtcl-fixed-alpha")
    if spec.qtcl_blocks > 1:
        command.extend(["--qtcl2-alpha", str(args.qtcl_alpha), "--qtcl2-fixed-alpha"])

    return command


def _normalize_for_compare(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_for_compare(item) for item in value]
    if isinstance(value, list):
        return [_normalize_for_compare(item) for item in value]
    return value


def expected_run_config(args: argparse.Namespace, spec: MethodSpec, fraction: float, run_output_dir: Path) -> dict[str, Any]:
    expected = {
        "data_dir": str(args.data_dir),
        "output_dir": str(run_output_dir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "optimizer": str(args.optimizer),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "pretrained": bool(args.pretrained),
        "init_checkpoint": None,
        "train_classifier_only": False,
        "train_final_layer_only": False,
        "model": spec.model,
        "train_fraction": float(fraction),
        "val_fraction": float(FULL_VAL_FRACTION),
        "seed": int(args.seed),
        "seeds": [int(args.seed)],
        "lr_step_size": int(args.lr_step_size),
        "lr_gamma": float(args.lr_gamma),
        "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
    }

    if spec.key in {"tcl12", "qtcl12"}:
        expected.update(
            {
                "tcl1_out_channels": int(args.tcl1_out_channels),
                "tcl1_out_height": int(args.tcl1_out_height),
                "tcl1_out_width": int(args.tcl1_out_width),
                "tcl2_out_channels": int(args.tcl2_out_channels),
                "tcl2_out_height": int(args.tcl2_out_height),
                "tcl2_out_width": int(args.tcl2_out_width),
            }
        )

    if spec.key == "qtcl12":
        expected.update(
            {
                "qtcl_n_qubits": int(args.qtcl_n_qubits),
                "qtcl_n_layers": int(args.qtcl_n_layers),
                "qtcl_F": args.qtcl_F,
                "qtcl_alpha": float(args.qtcl_alpha),
                "qtcl_learnable_alpha": False,
                "qtcl_shots": int(args.qtcl_shots),
                "qtcl_ansatz": str(args.qtcl_ansatz),
                "qtcl_ansatz_kwargs": str(args.qtcl_ansatz_kwargs),
                "qtcl2_n_qubits": args.qtcl2_n_qubits,
                "qtcl2_n_layers": args.qtcl2_n_layers,
                "qtcl2_F": args.qtcl2_F,
                "qtcl2_alpha": float(args.qtcl_alpha),
                "qtcl2_learnable_alpha": False,
                "qtcl2_shots": args.qtcl2_shots,
                "qtcl2_ansatz": args.qtcl2_ansatz,
                "qtcl2_ansatz_kwargs": args.qtcl2_ansatz_kwargs,
                "qtcl_lr": float(args.qtcl_lr),
                "qtcl_optimizer": str(args.qtcl_optimizer),
                "qtcl_weight_decay": float(args.qtcl_weight_decay),
                "qtcl_betas": [float(args.qtcl_betas[0]), float(args.qtcl_betas[1])],
                "qtcl_batchnorm": bool(args.qtcl_batchnorm),
            }
        )

    return expected


def existing_run_mismatches(
    args: argparse.Namespace,
    spec: MethodSpec,
    fraction: float,
    run_output_dir: Path,
) -> list[str]:
    config_path = run_output_dir / "experiment_config.json"
    if not config_path.exists():
        return [f"missing {config_path.name}"]

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    observed_args = payload.get("args")
    if not isinstance(observed_args, dict):
        return ["invalid experiment_config.json args payload"]

    expected = expected_run_config(args, spec, fraction, run_output_dir)
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        observed_value = observed_args.get(key)
        if _normalize_for_compare(observed_value) != _normalize_for_compare(expected_value):
            mismatches.append(f"{key}: expected {expected_value!r}, found {observed_value!r}")
    return mismatches


def extract_space_saving(run_log_path: Path) -> float | None:
    if not run_log_path.exists():
        return None

    matches = re.findall(r"space_saving=([0-9]*\.[0-9]+)", run_log_path.read_text(encoding="utf-8"))
    if not matches:
        return None
    return float(matches[-1])


def verify_qtcl12_space_saving(run_output_dir: Path) -> float:
    run_log_path = run_output_dir / "run.log"
    observed = extract_space_saving(run_log_path)
    if observed is None:
        raise RuntimeError(f"Could not find a space_saving entry in {run_log_path}.")
    if abs(observed - EXPECTED_QTCL12_SPACE_SAVING) > EXPECTED_QTCL12_SPACE_SAVING_TOL:
        raise RuntimeError(
            "qtcl12 space_saving mismatch "
            f"(expected approximately {EXPECTED_QTCL12_SPACE_SAVING:.4f}, found {observed:.4f})."
        )
    return observed


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"No rows were found in {path}.")
    return row


def load_training_summary(summary_path: Path) -> dict[str, Any]:
    row = read_single_row_csv(summary_path)
    int_fields = {"best_val_acc_epoch", "best_val_loss_epoch"}
    converted: dict[str, Any] = {}
    for key, value in row.items():
        if key in int_fields:
            converted[key] = int(float(value))
        else:
            converted[key] = float(value)
    return converted


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_sweep_results(output_dir: Path, results: Iterable[dict[str, Any]]) -> Path:
    path = output_dir / "sweep_results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELD_ORDER)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return path


def write_metric_pivot(
    output_dir: Path,
    results: list[dict[str, Any]],
    metric: str,
    methods: list[str],
    fractions: list[float],
) -> Path:
    if metric not in {"best_val_acc", "final_val_acc"}:
        raise ValueError(f"Unsupported pivot metric: {metric}")

    path = output_dir / f"{metric}_pivot.csv"
    lookup = {
        (fraction_label(float(row["train_fraction"])), str(row["method"])): row[metric]
        for row in results
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["train_fraction", *methods])
        for fraction in fractions:
            fraction_key = fraction_label(fraction)
            writer.writerow(
                [
                    fraction_key,
                    *[
                        f"{lookup[(fraction_key, method)] * 100:.4f}"
                        if (fraction_key, method) in lookup
                        else ""
                        for method in methods
                    ],
                ]
            )
    return path


def write_sweep_config(output_dir: Path, args: argparse.Namespace) -> Path:
    payload = {
        "training_module": TRAINING_MODULE,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "fractions": [float(fraction) for fraction in args.fractions],
        "methods": list(args.methods),
        "initialization": "imagenet_pretrained" if args.pretrained else "random",
        "seed": int(args.seed),
        "val_fraction": FULL_VAL_FRACTION,
        "expected_qtcl12_space_saving": EXPECTED_QTCL12_SPACE_SAVING,
        "args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "pretrained": args.pretrained,
            "train_classifier_only": False,
            "train_final_layer_only": False,
            "tcl1_out_channels": args.tcl1_out_channels,
            "tcl1_out_height": args.tcl1_out_height,
            "tcl1_out_width": args.tcl1_out_width,
            "tcl2_out_channels": args.tcl2_out_channels,
            "tcl2_out_height": args.tcl2_out_height,
            "tcl2_out_width": args.tcl2_out_width,
            "qtcl_n_qubits": args.qtcl_n_qubits,
            "qtcl_n_layers": args.qtcl_n_layers,
            "qtcl_F": args.qtcl_F,
            "qtcl_alpha": args.qtcl_alpha,
            "qtcl_shots": args.qtcl_shots,
            "qtcl_ansatz": args.qtcl_ansatz,
            "qtcl_ansatz_kwargs": args.qtcl_ansatz_kwargs,
            "qtcl2_n_qubits": args.qtcl2_n_qubits,
            "qtcl2_n_layers": args.qtcl2_n_layers,
            "qtcl2_F": args.qtcl2_F,
            "qtcl2_shots": args.qtcl2_shots,
            "qtcl2_ansatz": args.qtcl2_ansatz,
            "qtcl2_ansatz_kwargs": args.qtcl2_ansatz_kwargs,
            "qtcl_lr": args.qtcl_lr,
            "qtcl_optimizer": args.qtcl_optimizer,
            "qtcl_weight_decay": args.qtcl_weight_decay,
            "qtcl_betas": list(args.qtcl_betas),
            "qtcl_batchnorm": args.qtcl_batchnorm,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "no_progress": args.no_progress,
            "rerun": args.rerun,
        },
    }
    path = output_dir / "sweep_config.json"
    write_json(path, payload)
    return path


def collect_result(
    summary_path: Path,
    spec: MethodSpec,
    fraction: float,
    seed: int,
    run_output_dir: Path,
    qtcl_alpha: float,
    initialization: str,
) -> dict[str, Any]:
    summary = load_training_summary(summary_path)
    return {
        "method": spec.key,
        "model": spec.model,
        "train_fraction": float(fraction),
        "val_fraction": FULL_VAL_FRACTION,
        "seed": int(seed),
        "qtcl_alpha": qtcl_alpha if spec.qtcl_blocks > 0 else "",
        "qtcl2_alpha": qtcl_alpha if spec.qtcl_blocks > 1 else "",
        "initialization": initialization,
        "run_output_dir": str(run_output_dir),
        **summary,
    }


def run_one(
    args: argparse.Namespace,
    spec: MethodSpec,
    fraction: float,
    run_output_dir: Path,
) -> Path:
    summary_path = run_output_dir / "training_summary.csv"
    if summary_path.exists() and not args.rerun:
        mismatches = existing_run_mismatches(args, spec, fraction, run_output_dir)
        if not mismatches and spec.key == "qtcl12":
            observed_space_saving = extract_space_saving(run_output_dir / "run.log")
            if observed_space_saving is None:
                mismatches.append("missing qtcl12 space_saving log entry")
            elif abs(observed_space_saving - EXPECTED_QTCL12_SPACE_SAVING) > EXPECTED_QTCL12_SPACE_SAVING_TOL:
                mismatches.append(
                    "qtcl12 space_saving drifted "
                    f"(expected approximately {EXPECTED_QTCL12_SPACE_SAVING:.4f}, "
                    f"found {observed_space_saving:.4f})"
                )

        if not mismatches:
            print(
                f"[skip] method={spec.key} | train_fraction={fraction_label(fraction)} "
                f"| using existing summary at {summary_path}"
            )
            return summary_path

        print(
            f"[rerun] method={spec.key} | train_fraction={fraction_label(fraction)} "
            f"| reason={mismatches[0]}"
        )

    run_output_dir.mkdir(parents=True, exist_ok=True)
    command = child_command(args, spec, fraction, run_output_dir)
    print(f"[run ] method={spec.key} | train_fraction={fraction_label(fraction)} | seed={args.seed}")
    print(f"       {shlex.join(command)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected training summary was not created: {summary_path}")
    if spec.key == "qtcl12":
        observed_space_saving = verify_qtcl12_space_saving(run_output_dir)
        print(f"       verified qtcl12 space_saving={observed_space_saving:.4f}")
    return summary_path


def print_final_summary(results: list[dict[str, Any]], methods: list[str], fractions: list[float]) -> None:
    print("\nBest validation accuracy (%) by method and train fraction:")
    for fraction in fractions:
        fraction_key = fraction_label(fraction)
        metrics = []
        for method in methods:
            match = next(
                (
                    row
                    for row in results
                    if fraction_label(float(row["train_fraction"])) == fraction_key and row["method"] == method
                ),
                None,
            )
            if match is None:
                continue
            metrics.append(f"{method}={match['best_val_acc'] * 100:.2f}")
        print(f"  fraction={fraction_key} | " + " | ".join(metrics))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_sweep_config(args.output_dir, args)
    print(f"Sweep configuration written to: {config_path}")
    print(f"Validation fraction is fixed to {fraction_label(FULL_VAL_FRACTION)} for every run.")
    print(f"Initialization: {'ImageNet pretrained' if args.pretrained else 'random'}")
    print("Training scope: full network fine-tuning.")

    results: list[dict[str, Any]] = []
    for fraction in args.fractions:
        for method in args.methods:
            spec = METHOD_SPECS[method]
            run_output_dir = build_run_output_dir(args.output_dir, method, fraction)
            summary_path = run_one(args, spec, fraction, run_output_dir)
            result = collect_result(
                summary_path=summary_path,
                spec=spec,
                fraction=fraction,
                seed=args.seed,
                run_output_dir=run_output_dir,
                qtcl_alpha=args.qtcl_alpha,
                initialization="imagenet_pretrained" if args.pretrained else "random",
            )
            results.append(result)
            write_sweep_results(args.output_dir, results)
            write_metric_pivot(args.output_dir, results, "best_val_acc", list(args.methods), list(args.fractions))
            write_metric_pivot(args.output_dir, results, "final_val_acc", list(args.methods), list(args.fractions))

    print_final_summary(results, list(args.methods), list(args.fractions))
    print(f"\nSweep summary saved to: {args.output_dir / 'sweep_results.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Sweep interrupted by user.")
        raise SystemExit(130)
