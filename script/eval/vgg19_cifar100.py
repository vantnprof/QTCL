from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.exp import vgg19_cifar100 as train_lib


DEFAULTS = {
    "data_dir": Path("./data"),
    "batch_size": 128,
    "num_workers": 8,
    "train_fraction": 1.0,
    "val_fraction": 1.0,
    "seed": 42,
    "model": "vgg19",
    "tcl1_out_channels": 64,
    "tcl1_out_height": 8,
    "tcl1_out_width": 8,
    "tcl2_out_channels": 32,
    "tcl2_out_height": 8,
    "tcl2_out_width": 8,
    "qtcl_n_qubits": 8,
    "qtcl_n_layers": 2,
    "qtcl_F": None,
    "qtcl_alpha": 0.5,
    "qtcl_learnable_alpha": False,
    "qtcl_shots": 0,
    "qtcl_ansatz": "HEA",
    "qtcl_ansatz_kwargs": "{}",
    "qtcl2_n_qubits": None,
    "qtcl2_n_layers": None,
    "qtcl2_F": None,
    "qtcl2_alpha": None,
    "qtcl2_learnable_alpha": None,
    "qtcl2_shots": None,
    "qtcl2_ansatz": None,
    "qtcl2_ansatz_kwargs": None,
    "qtcl_batchnorm": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VGG19 CIFAR-100 checkpoint on the validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the saved checkpoint.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root directory. Defaults to the saved training value from the checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size used to rebuild the validation dataloader. Defaults to the checkpoint value.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Dataloader worker count. Defaults to the checkpoint value.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Optional override for the validation subset fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the validation subset seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on, for example cpu, cuda, or cuda:0. Defaults to CUDA when available.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def checkpoint_arg(checkpoint_args: Mapping[str, Any], key: str) -> Any:
    value = checkpoint_args.get(key, DEFAULTS[key])
    return DEFAULTS[key] if value is None and key in DEFAULTS else value


def resolve_repo_path(path_like: Any) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_arg!r} was requested, but CUDA is not available.")
    return device


def parse_ansatz_kwargs(raw_value: Any, arg_name: str) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if not isinstance(raw_value, str):
        raise ValueError(f"{arg_name} must be stored as a JSON string or dict, got {type(raw_value).__name__}.")
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {arg_name} as JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object.")
    return dict(parsed)


def canonical_model_variant(model_name: str) -> str:
    if model_name == "vgg19_tcl":
        return "vgg19_tcl1"
    if model_name == "vgg19_qtcl":
        return "vgg19_qtcl1"
    return model_name


def build_model_from_checkpoint(checkpoint_args: Mapping[str, Any], device: torch.device) -> tuple[nn.Module, str]:
    model = train_lib.build_model(num_classes=100, pretrained=False, device=device)
    model_variant = canonical_model_variant(str(checkpoint_arg(checkpoint_args, "model")))

    qtcl_first_kwargs: Optional[dict[str, Any]] = None
    qtcl_second_kwargs: Optional[dict[str, Any]] = None
    if model_variant in {"vgg19_qtcl1", "vgg19_qtcl12"}:
        qtcl_first_kwargs = {
            "n_qubits": int(checkpoint_arg(checkpoint_args, "qtcl_n_qubits")),
            "n_layers": int(checkpoint_arg(checkpoint_args, "qtcl_n_layers")),
            "F": checkpoint_arg(checkpoint_args, "qtcl_F"),
            "shots": None
            if int(checkpoint_arg(checkpoint_args, "qtcl_shots")) == 0
            else int(checkpoint_arg(checkpoint_args, "qtcl_shots")),
            "ansatz_type": str(checkpoint_arg(checkpoint_args, "qtcl_ansatz")),
            "ansatz_kwargs": parse_ansatz_kwargs(
                checkpoint_arg(checkpoint_args, "qtcl_ansatz_kwargs"),
                "qtcl_ansatz_kwargs",
            ),
            "alpha": float(checkpoint_arg(checkpoint_args, "qtcl_alpha")),
            "learnable_alpha": bool(checkpoint_arg(checkpoint_args, "qtcl_learnable_alpha")),
        }
        qtcl_second_kwargs = copy.deepcopy(qtcl_first_kwargs)

        if checkpoint_arg(checkpoint_args, "qtcl2_n_qubits") is not None:
            qtcl_second_kwargs["n_qubits"] = int(checkpoint_arg(checkpoint_args, "qtcl2_n_qubits"))
        if checkpoint_arg(checkpoint_args, "qtcl2_n_layers") is not None:
            qtcl_second_kwargs["n_layers"] = int(checkpoint_arg(checkpoint_args, "qtcl2_n_layers"))
        if checkpoint_arg(checkpoint_args, "qtcl2_F") is not None:
            qtcl_second_kwargs["F"] = int(checkpoint_arg(checkpoint_args, "qtcl2_F"))
        if checkpoint_arg(checkpoint_args, "qtcl2_shots") is not None:
            qtcl2_shots = int(checkpoint_arg(checkpoint_args, "qtcl2_shots"))
            qtcl_second_kwargs["shots"] = None if qtcl2_shots == 0 else qtcl2_shots
        if checkpoint_arg(checkpoint_args, "qtcl2_ansatz") is not None:
            qtcl_second_kwargs["ansatz_type"] = str(checkpoint_arg(checkpoint_args, "qtcl2_ansatz"))
        if checkpoint_arg(checkpoint_args, "qtcl2_ansatz_kwargs") is not None:
            qtcl_second_kwargs["ansatz_kwargs"] = parse_ansatz_kwargs(
                checkpoint_arg(checkpoint_args, "qtcl2_ansatz_kwargs"),
                "qtcl2_ansatz_kwargs",
            )
        if checkpoint_arg(checkpoint_args, "qtcl2_alpha") is not None:
            qtcl_second_kwargs["alpha"] = float(checkpoint_arg(checkpoint_args, "qtcl2_alpha"))
        if checkpoint_arg(checkpoint_args, "qtcl2_learnable_alpha") is not None:
            qtcl_second_kwargs["learnable_alpha"] = bool(checkpoint_arg(checkpoint_args, "qtcl2_learnable_alpha"))

    if model_variant == "vgg19_tcl1":
        model = train_lib.replace_1st_fc_with_tcl(
            model,
            out_channels=int(checkpoint_arg(checkpoint_args, "tcl1_out_channels")),
            out_height=int(checkpoint_arg(checkpoint_args, "tcl1_out_height")),
            out_width=int(checkpoint_arg(checkpoint_args, "tcl1_out_width")),
            debug=False,
        )
    elif model_variant == "vgg19_tcl12":
        model = train_lib.replace_1st_2nd_fc_with_tcl(
            model,
            first_out_channels=int(checkpoint_arg(checkpoint_args, "tcl1_out_channels")),
            first_out_height=int(checkpoint_arg(checkpoint_args, "tcl1_out_height")),
            first_out_width=int(checkpoint_arg(checkpoint_args, "tcl1_out_width")),
            second_out_channels=int(checkpoint_arg(checkpoint_args, "tcl2_out_channels")),
            second_out_height=int(checkpoint_arg(checkpoint_args, "tcl2_out_height")),
            second_out_width=int(checkpoint_arg(checkpoint_args, "tcl2_out_width")),
            debug=False,
        )
    elif model_variant == "vgg19_qtcl1":
        model = train_lib.replace_1st_fc_with_qtcl(
            model,
            out_channels=int(checkpoint_arg(checkpoint_args, "tcl1_out_channels")),
            out_height=int(checkpoint_arg(checkpoint_args, "tcl1_out_height")),
            out_width=int(checkpoint_arg(checkpoint_args, "tcl1_out_width")),
            use_batchnorm=True,
            qmtl_kwargs=qtcl_first_kwargs,
            debug=False,
        )
    elif model_variant == "vgg19_qtcl12":
        model = train_lib.replace_1st_2nd_fc_with_qtcl(
            model,
            first_out_channels=int(checkpoint_arg(checkpoint_args, "tcl1_out_channels")),
            first_out_height=int(checkpoint_arg(checkpoint_args, "tcl1_out_height")),
            first_out_width=int(checkpoint_arg(checkpoint_args, "tcl1_out_width")),
            second_out_channels=int(checkpoint_arg(checkpoint_args, "tcl2_out_channels")),
            second_out_height=int(checkpoint_arg(checkpoint_args, "tcl2_out_height")),
            second_out_width=int(checkpoint_arg(checkpoint_args, "tcl2_out_width")),
            use_batchnorm=bool(checkpoint_arg(checkpoint_args, "qtcl_batchnorm")),
            first_qmtl_kwargs=qtcl_first_kwargs,
            second_qmtl_kwargs=qtcl_second_kwargs,
            debug=False,
        )
    elif model_variant != "vgg19":
        raise ValueError(f"Unsupported VGG19 model variant in checkpoint: {model_variant}")

    return model, model_variant


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must deserialize to a mapping.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing the 'model_state_dict' entry.")

    checkpoint_args_raw = checkpoint.get("args")
    if checkpoint_args_raw is None:
        raise ValueError("Checkpoint is missing the saved training arguments under 'args'.")
    if not isinstance(checkpoint_args_raw, Mapping):
        raise ValueError("Checkpoint entry 'args' must be a mapping.")
    checkpoint_args = dict(checkpoint_args_raw)

    device = select_device(args.device)
    data_dir = resolve_repo_path(args.data_dir if args.data_dir is not None else checkpoint_arg(checkpoint_args, "data_dir"))
    batch_size = int(args.batch_size if args.batch_size is not None else checkpoint_arg(checkpoint_args, "batch_size"))
    num_workers = int(
        args.num_workers if args.num_workers is not None else checkpoint_arg(checkpoint_args, "num_workers")
    )
    train_fraction = float(checkpoint_arg(checkpoint_args, "train_fraction"))
    val_fraction = float(
        args.val_fraction if args.val_fraction is not None else checkpoint_arg(checkpoint_args, "val_fraction")
    )
    seed = int(args.seed if args.seed is not None else checkpoint_arg(checkpoint_args, "seed"))

    model, model_variant = build_model_from_checkpoint(checkpoint_args, device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict did not load cleanly "
            f"(missing={missing_keys}, unexpected={unexpected_keys})."
        )

    _, val_loader, _, _ = train_lib.build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        workers=num_workers,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
        distributed=False,
        rank=0,
        world_size=1,
    )

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = train_lib.evaluate(
        model,
        val_loader,
        criterion,
        device,
        epoch_desc="Validation",
        show_progress=not args.no_progress,
        distributed=False,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Model variant: {model_variant}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    if "best_val_acc" in checkpoint:
        print(f"Checkpoint best_val_acc: {float(checkpoint['best_val_acc']) * 100:.2f}%")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {int(checkpoint['epoch'])}")


if __name__ == "__main__":
    main()
