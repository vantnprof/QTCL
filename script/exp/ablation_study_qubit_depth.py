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
MODEL_VARIANT = "vgg19_qtcl12"
FULL_TRAIN_FRACTION = 1.0
FULL_VAL_FRACTION = 1.0
DEFAULT_QUBITS = (4, 8, 16, 32)
DEFAULT_DEPTHS = (1, 2, 3)
DEFAULT_INIT_CHECKPOINT = Path("./results/vgg19_qtcl12_alpha0.5/checkpoint_best_acc.pth")
DEFAULT_TRAIN_LAST_CLASSIFIER_FC_LAYERS = 3
EXPECTED_REFERENCE_SPACE_SAVING = 0.9868
EXPECTED_REFERENCE_SPACE_SAVING_TOL = 5e-4


@dataclass(frozen=True)
class SweepPoint:
    n_qubits: int
    depth: int


REFERENCE_POINT = SweepPoint(8, 2)


SUMMARY_FIELD_ORDER = [
    "n_qubits",
    "depth",
    "seed",
    "model",
    "train_fraction",
    "val_fraction",
    "qtcl_alpha",
    "qtcl2_alpha",
    "qtcl_ansatz",
    "qtcl2_ansatz",
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


def validate_json_object(raw_value: str, arg_name: str) -> None:
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must decode as a JSON object.") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{arg_name} must decode as a JSON object.")


def validate_positive_unique_sequence(values: list[int], arg_name: str) -> None:
    if not values:
        raise ValueError(f"{arg_name} must contain at least one value.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Every value in {arg_name} must be a positive integer.")
    if len(set(values)) != len(values):
        raise ValueError(f"{arg_name} must not contain duplicate values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train VGG19_QTCL12 on the full CIFAR100 training split and evaluate on the full validation split "
            "while sweeping QTCL qubit count and circuit depth."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Dataset root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/ablation_study_qubit_depth"),
        help="Root directory for all per-run artifacts and sweep summaries.",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=list(DEFAULT_QUBITS),
        help="QTCL qubit counts to evaluate for both QTCL blocks.",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=None,
        help="Convenience override for running a single qubit count; equivalent to --qubits NQ.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEPTHS),
        help="QTCL circuit depths (number of layers) to evaluate for both QTCL blocks.",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=None,
        help="Convenience override for running a single circuit depth; equivalent to --depths P.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Single seed to use for every run.")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate for the SGD optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for SGD.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained VGG19 weights.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=DEFAULT_INIT_CHECKPOINT,
        help="Checkpoint used to initialize compatible weights before each ablation run.",
    )
    parser.add_argument(
        "--train-last-classifier-fc-layers",
        type=int,
        default=DEFAULT_TRAIN_LAST_CLASSIFIER_FC_LAYERS,
        help="Keep only the last N classifier FC-like layers trainable during the sweep.",
    )
    parser.add_argument(
        "--tcl1-out-channels",
        type=int,
        default=512,
        help="Output channels for the first QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl1-out-height",
        type=int,
        default=3,
        help="Output height for the first QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl1-out-width",
        type=int,
        default=3,
        help="Output width for the first QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl2-out-channels",
        type=int,
        default=512,
        help="Output channels for the second QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl2-out-height",
        type=int,
        default=3,
        help="Output height for the second QTCL replacement block.",
    )
    parser.add_argument(
        "--tcl2-out-width",
        type=int,
        default=3,
        help="Output width for the second QTCL replacement block.",
    )
    parser.add_argument(
        "--qtcl-F",
        type=int,
        default=None,
        help="Optional latent feature width F for the first QTCL block. Defaults to n_qubits when omitted.",
    )
    parser.add_argument(
        "--qtcl2-F",
        type=int,
        default=None,
        help="Optional latent feature width F override for the second QTCL block.",
    )
    parser.add_argument("--qtcl-alpha", type=float, default=0.5, help="Fixed alpha for both QTCL blocks.")
    parser.add_argument(
        "--qtcl-shots",
        type=int,
        default=0,
        help="Number of quantum shots for the first QTCL block (0 means analytic expectation).",
    )
    parser.add_argument(
        "--qtcl2-shots",
        type=int,
        default=None,
        help="Optional shots override for the second QTCL block (defaults to --qtcl-shots).",
    )
    parser.add_argument("--qtcl-ansatz", type=str, default="HEA", help="Ansatz type for the first QTCL block.")
    parser.add_argument(
        "--qtcl-ansatz-kwargs",
        type=str,
        default="{}",
        help="JSON object of extra kwargs for the first QTCL ansatz.",
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
    if args.nq is not None:
        args.qubits = [args.nq]
    if args.p is not None:
        args.depths = [args.p]
    args.qtcl_betas = tuple(args.qtcl_betas)
    args.data_dir = resolve_repo_path(Path(args.data_dir))
    args.output_dir = resolve_repo_path(Path(args.output_dir))
    args.init_checkpoint = resolve_repo_path(Path(args.init_checkpoint))
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")
    if not args.init_checkpoint.exists():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {args.init_checkpoint}")
    if args.train_last_classifier_fc_layers <= 0:
        raise ValueError("train-last-classifier-fc-layers must be a positive integer.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")
    if args.momentum < 0.0:
        raise ValueError("momentum must be non-negative.")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative.")
    if args.lr_step_size <= 0:
        raise ValueError("lr-step-size must be a positive integer.")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("lr-gamma must lie in the interval (0, 1].")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze-backbone-epochs must be non-negative.")
    if args.qtcl_F is not None and args.qtcl_F <= 0:
        raise ValueError("qtcl-F must be a positive integer when provided.")
    if args.qtcl2_F is not None and args.qtcl2_F <= 0:
        raise ValueError("qtcl2-F must be a positive integer when provided.")
    if args.qtcl_shots < 0:
        raise ValueError("qtcl-shots must be non-negative.")
    if args.qtcl2_shots is not None and args.qtcl2_shots < 0:
        raise ValueError("qtcl2-shots must be non-negative when provided.")
    if not 0.0 <= args.qtcl_alpha <= 1.0:
        raise ValueError("qtcl-alpha must lie in the interval [0, 1].")
    if args.tcl1_out_channels <= 0 or args.tcl1_out_height <= 0 or args.tcl1_out_width <= 0:
        raise ValueError("First replacement dimensions must all be positive.")
    if args.tcl2_out_channels <= 0 or args.tcl2_out_height <= 0 or args.tcl2_out_width <= 0:
        raise ValueError("Second replacement dimensions must all be positive.")

    validate_positive_unique_sequence(list(args.qubits), "--qubits")
    validate_positive_unique_sequence(list(args.depths), "--depths")
    validate_json_object(args.qtcl_ansatz_kwargs, "--qtcl-ansatz-kwargs")
    if args.qtcl2_ansatz_kwargs is not None:
        validate_json_object(args.qtcl2_ansatz_kwargs, "--qtcl2-ansatz-kwargs")


def effective_qtcl2_shots(args: argparse.Namespace) -> int:
    return args.qtcl2_shots if args.qtcl2_shots is not None else args.qtcl_shots


def effective_qtcl2_ansatz(args: argparse.Namespace) -> str:
    return args.qtcl2_ansatz if args.qtcl2_ansatz is not None else args.qtcl_ansatz


def effective_qtcl2_ansatz_kwargs(args: argparse.Namespace) -> str:
    return args.qtcl2_ansatz_kwargs if args.qtcl2_ansatz_kwargs is not None else args.qtcl_ansatz_kwargs


def effective_qtcl2_F(args: argparse.Namespace) -> int | None:
    return args.qtcl2_F if args.qtcl2_F is not None else args.qtcl_F


def build_run_output_dir(output_dir: Path, point: SweepPoint) -> Path:
    return output_dir / f"nq_{point.n_qubits}" / f"p_{point.depth}"


def child_command(args: argparse.Namespace, point: SweepPoint, run_output_dir: Path) -> list[str]:
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
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--model",
        MODEL_VARIANT,
        "--init-checkpoint",
        str(args.init_checkpoint),
        "--allow-partial-init",
        "--train-last-classifier-fc-layers",
        str(args.train_last_classifier_fc_layers),
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
        str(point.n_qubits),
        "--qtcl-n-layers",
        str(point.depth),
        "--qtcl-alpha",
        str(args.qtcl_alpha),
        "--qtcl-fixed-alpha",
        "--qtcl-shots",
        str(args.qtcl_shots),
        "--qtcl-ansatz",
        args.qtcl_ansatz,
        "--qtcl-ansatz-kwargs",
        args.qtcl_ansatz_kwargs,
        "--qtcl2-n-qubits",
        str(point.n_qubits),
        "--qtcl2-n-layers",
        str(point.depth),
        "--qtcl2-alpha",
        str(args.qtcl_alpha),
        "--qtcl2-fixed-alpha",
        "--qtcl2-shots",
        str(effective_qtcl2_shots(args)),
        "--qtcl2-ansatz",
        effective_qtcl2_ansatz(args),
        "--qtcl2-ansatz-kwargs",
        effective_qtcl2_ansatz_kwargs(args),
        "--qtcl-lr",
        str(args.qtcl_lr),
        "--qtcl-weight-decay",
        str(args.qtcl_weight_decay),
        "--qtcl-betas",
        str(args.qtcl_betas[0]),
        str(args.qtcl_betas[1]),
        "--train-fraction",
        str(FULL_TRAIN_FRACTION),
        "--val-fraction",
        str(FULL_VAL_FRACTION),
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
    qtcl2_F = effective_qtcl2_F(args)
    if qtcl2_F is not None:
        command.extend(["--qtcl2-F", str(qtcl2_F)])

    return command


def _normalize_for_compare(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_for_compare(item) for item in value]
    if isinstance(value, list):
        return [_normalize_for_compare(item) for item in value]
    return value


def expected_run_config(args: argparse.Namespace, point: SweepPoint, run_output_dir: Path) -> dict[str, Any]:
    return {
        "data_dir": str(args.data_dir),
        "output_dir": str(run_output_dir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "pretrained": bool(args.pretrained),
        "model": MODEL_VARIANT,
        "init_checkpoint": str(args.init_checkpoint),
        "allow_partial_init": True,
        "train_last_classifier_fc_layers": int(args.train_last_classifier_fc_layers),
        "seed": int(args.seed),
        "seeds": [int(args.seed)],
        "train_fraction": float(FULL_TRAIN_FRACTION),
        "val_fraction": float(FULL_VAL_FRACTION),
        "tcl1_out_channels": int(args.tcl1_out_channels),
        "tcl1_out_height": int(args.tcl1_out_height),
        "tcl1_out_width": int(args.tcl1_out_width),
        "tcl2_out_channels": int(args.tcl2_out_channels),
        "tcl2_out_height": int(args.tcl2_out_height),
        "tcl2_out_width": int(args.tcl2_out_width),
        "qtcl_n_qubits": int(point.n_qubits),
        "qtcl_n_layers": int(point.depth),
        "qtcl_F": args.qtcl_F,
        "qtcl_alpha": float(args.qtcl_alpha),
        "qtcl_learnable_alpha": False,
        "qtcl_shots": int(args.qtcl_shots),
        "qtcl_ansatz": str(args.qtcl_ansatz),
        "qtcl_ansatz_kwargs": str(args.qtcl_ansatz_kwargs),
        "qtcl2_n_qubits": int(point.n_qubits),
        "qtcl2_n_layers": int(point.depth),
        "qtcl2_F": effective_qtcl2_F(args),
        "qtcl2_alpha": float(args.qtcl_alpha),
        "qtcl2_learnable_alpha": False,
        "qtcl2_shots": int(effective_qtcl2_shots(args)),
        "qtcl2_ansatz": str(effective_qtcl2_ansatz(args)),
        "qtcl2_ansatz_kwargs": str(effective_qtcl2_ansatz_kwargs(args)),
        "qtcl_lr": float(args.qtcl_lr),
        "qtcl_weight_decay": float(args.qtcl_weight_decay),
        "qtcl_betas": [float(args.qtcl_betas[0]), float(args.qtcl_betas[1])],
        "qtcl_batchnorm": bool(args.qtcl_batchnorm),
        "lr_step_size": int(args.lr_step_size),
        "lr_gamma": float(args.lr_gamma),
        "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
    }


def existing_run_mismatches(args: argparse.Namespace, point: SweepPoint, run_output_dir: Path) -> list[str]:
    config_path = run_output_dir / "experiment_config.json"
    if not config_path.exists():
        return [f"missing {config_path.name}"]

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    observed_args = payload.get("args")
    if not isinstance(observed_args, dict):
        return ["invalid experiment_config.json args payload"]

    expected = expected_run_config(args, point, run_output_dir)
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


def expected_space_saving_for_point(point: SweepPoint) -> float | None:
    if point == REFERENCE_POINT:
        return EXPECTED_REFERENCE_SPACE_SAVING
    return None


def verify_space_saving(run_output_dir: Path, point: SweepPoint) -> float | None:
    expected = expected_space_saving_for_point(point)
    if expected is None:
        return None

    run_log_path = run_output_dir / "run.log"
    observed = extract_space_saving(run_log_path)
    if observed is None:
        raise RuntimeError(f"Could not find a space_saving entry in {run_log_path}.")
    if abs(observed - expected) > EXPECTED_REFERENCE_SPACE_SAVING_TOL:
        raise RuntimeError(
            f"nq={point.n_qubits}, p={point.depth} space_saving mismatch "
            f"(expected approximately {expected:.4f}, found {observed:.4f})."
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
    qubits: list[int],
    depths: list[int],
) -> Path:
    if metric not in {"best_val_acc", "final_val_acc"}:
        raise ValueError(f"Unsupported pivot metric: {metric}")

    path = output_dir / f"{metric}_pivot.csv"
    lookup = {
        (int(row["n_qubits"]), int(row["depth"])): float(row[metric])
        for row in results
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_qubits", *[f"p_{depth}" for depth in depths]])
        for n_qubits in qubits:
            writer.writerow(
                [
                    n_qubits,
                    *[
                        f"{lookup[(n_qubits, depth)] * 100:.4f}" if (n_qubits, depth) in lookup else ""
                        for depth in depths
                    ],
                ]
            )
    return path


def write_sweep_config(output_dir: Path, args: argparse.Namespace) -> Path:
    payload = {
        "training_module": TRAINING_MODULE,
        "model": MODEL_VARIANT,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "qubits": list(args.qubits),
        "depths": list(args.depths),
        "seed": int(args.seed),
        "train_fraction": FULL_TRAIN_FRACTION,
        "val_fraction": FULL_VAL_FRACTION,
        "expected_reference_point": {
            "n_qubits": REFERENCE_POINT.n_qubits,
            "depth": REFERENCE_POINT.depth,
            "space_saving": EXPECTED_REFERENCE_SPACE_SAVING,
        },
        "args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "pretrained": args.pretrained,
            "init_checkpoint": str(args.init_checkpoint),
            "allow_partial_init": True,
            "train_last_classifier_fc_layers": args.train_last_classifier_fc_layers,
            "tcl1_out_channels": args.tcl1_out_channels,
            "tcl1_out_height": args.tcl1_out_height,
            "tcl1_out_width": args.tcl1_out_width,
            "tcl2_out_channels": args.tcl2_out_channels,
            "tcl2_out_height": args.tcl2_out_height,
            "tcl2_out_width": args.tcl2_out_width,
            "qtcl_F": args.qtcl_F,
            "qtcl2_F": args.qtcl2_F,
            "qtcl_alpha": args.qtcl_alpha,
            "qtcl_shots": args.qtcl_shots,
            "qtcl2_shots": args.qtcl2_shots,
            "qtcl_ansatz": args.qtcl_ansatz,
            "qtcl_ansatz_kwargs": args.qtcl_ansatz_kwargs,
            "qtcl2_ansatz": args.qtcl2_ansatz,
            "qtcl2_ansatz_kwargs": args.qtcl2_ansatz_kwargs,
            "effective_qtcl2_shots": effective_qtcl2_shots(args),
            "effective_qtcl2_ansatz": effective_qtcl2_ansatz(args),
            "effective_qtcl2_ansatz_kwargs": effective_qtcl2_ansatz_kwargs(args),
            "effective_qtcl2_F": effective_qtcl2_F(args),
            "qtcl_lr": args.qtcl_lr,
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
    point: SweepPoint,
    seed: int,
    run_output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = load_training_summary(summary_path)
    return {
        "n_qubits": point.n_qubits,
        "depth": point.depth,
        "seed": int(seed),
        "model": MODEL_VARIANT,
        "train_fraction": FULL_TRAIN_FRACTION,
        "val_fraction": FULL_VAL_FRACTION,
        "qtcl_alpha": args.qtcl_alpha,
        "qtcl2_alpha": args.qtcl_alpha,
        "qtcl_ansatz": args.qtcl_ansatz,
        "qtcl2_ansatz": effective_qtcl2_ansatz(args),
        "run_output_dir": str(run_output_dir),
        **summary,
    }


def run_one(args: argparse.Namespace, point: SweepPoint, run_output_dir: Path) -> Path:
    summary_path = run_output_dir / "training_summary.csv"
    if summary_path.exists() and not args.rerun:
        mismatches = existing_run_mismatches(args, point, run_output_dir)
        expected_space_saving = expected_space_saving_for_point(point)
        if not mismatches and expected_space_saving is not None:
            observed_space_saving = extract_space_saving(run_output_dir / "run.log")
            if observed_space_saving is None:
                mismatches.append("missing reference space_saving log entry")
            elif abs(observed_space_saving - expected_space_saving) > EXPECTED_REFERENCE_SPACE_SAVING_TOL:
                mismatches.append(
                    f"reference space_saving drifted (expected approximately {expected_space_saving:.4f}, "
                    f"found {observed_space_saving:.4f})"
                )

        if not mismatches:
            print(
                f"[skip] nq={point.n_qubits} | p={point.depth} "
                f"| using existing summary at {summary_path}"
            )
            return summary_path

        print(f"[rerun] nq={point.n_qubits} | p={point.depth} | reason={mismatches[0]}")

    run_output_dir.mkdir(parents=True, exist_ok=True)
    command = child_command(args, point, run_output_dir)
    print(f"[run ] nq={point.n_qubits} | p={point.depth} | seed={args.seed}")
    print(f"       {shlex.join(command)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected training summary was not created: {summary_path}")
    observed_space_saving = verify_space_saving(run_output_dir, point)
    if observed_space_saving is not None:
        print(f"       verified reference space_saving={observed_space_saving:.4f}")
    return summary_path


def print_final_summary(results: list[dict[str, Any]], qubits: list[int], depths: list[int]) -> None:
    print("\nBest validation accuracy (%) by qubit count and depth:")
    for n_qubits in qubits:
        metrics = []
        for depth in depths:
            match = next(
                (
                    row
                    for row in results
                    if int(row["n_qubits"]) == n_qubits and int(row["depth"]) == depth
                ),
                None,
            )
            if match is None:
                continue
            metrics.append(f"p={depth}:{float(match['best_val_acc']) * 100:.2f}")
        print(f"  nq={n_qubits} | " + " | ".join(metrics))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_sweep_config(args.output_dir, args)
    print(f"Sweep configuration written to: {config_path}")
    print(
        f"Model is fixed to {MODEL_VARIANT} | "
        f"train_fraction={FULL_TRAIN_FRACTION:.1f} | val_fraction={FULL_VAL_FRACTION:.1f}"
    )

    results: list[dict[str, Any]] = []
    for n_qubits in args.qubits:
        for depth in args.depths:
            point = SweepPoint(n_qubits=n_qubits, depth=depth)
            run_output_dir = build_run_output_dir(args.output_dir, point)
            summary_path = run_one(args, point, run_output_dir)
            result = collect_result(
                summary_path=summary_path,
                point=point,
                seed=args.seed,
                run_output_dir=run_output_dir,
                args=args,
            )
            results.append(result)
            write_sweep_results(args.output_dir, results)
            write_metric_pivot(args.output_dir, results, "best_val_acc", list(args.qubits), list(args.depths))
            write_metric_pivot(args.output_dir, results, "final_val_acc", list(args.qubits), list(args.depths))

    print_final_summary(results, list(args.qubits), list(args.depths))
    print(f"\nSweep summary saved to: {args.output_dir / 'sweep_results.csv'}")


if __name__ == "__main__":
    main()
