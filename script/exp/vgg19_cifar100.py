import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm
from src.tcl.tcl import TCL
from src.qtcl.qtcl import QTCL


class TeeStream:
    """Mirror writes to both console and log file."""

    def __init__(self, console_stream: Any, log_stream: Any) -> None:
        self._console_stream = console_stream
        self._log_stream = log_stream

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._console_stream.write(data)
        self._console_stream.flush()
        # tqdm emits frequent carriage-return refreshes. Mirroring every refresh
        # into the log file creates large logs and heavy I/O overhead, which can
        # noticeably slow long QTCL runs. Keep those transient updates on the
        # interactive console only; epoch summaries are still logged normally.
        if "\r" in data:
            return len(data)
        self._log_stream.write(data)
        return len(data)

    def flush(self) -> None:
        try:
            self._console_stream.flush()
        except Exception:
            pass
        try:
            self._log_stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return bool(getattr(self._console_stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self._console_stream, "encoding", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._console_stream, name)


def enable_run_log(output_dir: Path) -> Path:
    run_log_path = output_dir / "run.log"
    log_stream = run_log_path.open(mode="w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_stream)
    sys.stderr = TeeStream(sys.stderr, log_stream)
    print(f"Streaming console output to: {run_log_path}")
    return run_log_path


def replace_1st_fc_with_tcl(
    model: nn.Module,
    out_channels: int,
    out_height: int,
    out_width: int,
    *,
    debug: bool = False,
) -> nn.Module:
    if not hasattr(model, "classifier"):
        raise ValueError("Model does not expose a classifier attribute.")
    if not isinstance(model.classifier, nn.Sequential) or len(model.classifier) == 0:
        raise ValueError("Model classifier must be a non-empty nn.Sequential.")

    original_fc_params = sum(
        p.numel()
        for layer in model.classifier
        if isinstance(layer, nn.Linear)
        for p in layer.parameters()
        if p.requires_grad
    )

    first_layer = model.classifier[0]
    if not isinstance(first_layer, nn.Linear):
        raise ValueError("The first classifier layer is not a linear layer.")

    if not hasattr(model, "avgpool"):
        raise ValueError("Model does not expose an avgpool attribute.")

    output_size = getattr(model.avgpool, "output_size", None)
    if output_size is None:
        raise ValueError("Could not determine the output size of model.avgpool.")
    if isinstance(output_size, int):
        in_height, in_width = output_size, output_size
    else:
        in_height, in_width = output_size

    if in_height is None or in_width is None:
        raise ValueError("avgpool output size must resolve to concrete height and width.")

    in_features = first_layer.in_features
    if debug:
        print(
            "[replace_1st_fc_with_tcl] Detected first linear layer "
            f"in_features={in_features}, out_features={first_layer.out_features}"
        )

    expected_product = in_height * in_width
    if in_features % expected_product != 0:
        raise ValueError("First linear layer in_features is not divisible by avgpool spatial size.")

    in_channels = in_features // expected_product
    if debug:
        print(
            "[replace_1st_fc_with_tcl] AvgPool spatial size "
            f"({in_height}x{in_width}) -> inferred in_channels={in_channels}"
        )

    flattened_features = out_channels * out_height * out_width
    if debug:
        print(
            "[replace_1st_fc_with_tcl] Target TCL output "
            f"out_channels={out_channels}, out_height={out_height}, out_width={out_width}, "
            f"flattened_features={flattened_features}"
        )

    rest_layers = list(model.classifier[1:])

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if debug:
        print("[replace_1st_fc_with_tcl] Original classifier:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    # Adjust the following linear layer to accept the new flattened size.
    updated_linear = False
    for idx, layer in enumerate(rest_layers):
        if isinstance(layer, nn.Linear):
            if debug:
                print(
                    "[replace_1st_fc_with_tcl] Adjusting next linear layer "
                    f"from in_features={layer.in_features} to {flattened_features}"
                )
            rest_layers[idx] = nn.Linear(
                flattened_features,
                layer.out_features,
                bias=layer.bias is not None,
            ).to(device=device, dtype=dtype)
            updated_linear = True
            break
    if not updated_linear:
        raise ValueError("Could not locate the next linear layer in the classifier to adjust.")

    bn_in = nn.BatchNorm2d(in_channels).to(device=device, dtype=dtype)
    bn_out = nn.BatchNorm2d(out_channels).to(device=device, dtype=dtype)
    tcl_layer = TCL(
        in_channels=in_channels,
        out_channels=out_channels,
        out_height=out_height,
        out_width=out_width,
    ).to(device=device, dtype=dtype)

    new_classifier_layers = [
        nn.Unflatten(1, (in_channels, in_height, in_width)),
        bn_in,
        tcl_layer,
        bn_out,
        nn.Flatten(),
        *rest_layers,
    ]
    model.classifier = nn.Sequential(*new_classifier_layers)
    if debug:
        print("[replace_1st_fc_with_tcl] New classifier:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    tracked_layers = tuple(
        layer for layer in model.classifier if isinstance(layer, (nn.Linear, TCL, QTCL))
    )
    modified_fc_params = sum(
        p.numel()
        for layer in tracked_layers
        for p in layer.parameters()
        if p.requires_grad
    )
    if modified_fc_params <= 0 or original_fc_params <= 0:
        space_saving = float("nan")
    else:
        space_saving = 1.0 - (modified_fc_params / original_fc_params)

    print(
        "[replace_1st_fc_with_tcl] Parameter comparison "
        f"| original_fc={original_fc_params:,} "
        f"| modified_fc={modified_fc_params:,} "
        f"| space_saving={space_saving:.4f}"
    )
    return model


def replace_1st_2nd_fc_with_tcl(
    model: nn.Module,
    first_out_channels: int,
    first_out_height: int,
    first_out_width: int,
    second_out_channels: int,
    second_out_height: int,
    second_out_width: int,
    *,
    debug: bool = False,
) -> nn.Module:
    if not hasattr(model, "classifier"):
        raise ValueError("Model does not expose a classifier attribute.")
    if not isinstance(model.classifier, nn.Sequential) or len(model.classifier) == 0:
        raise ValueError("Model classifier must be a non-empty nn.Sequential.")

    original_fc_params = sum(
        p.numel()
        for layer in model.classifier
        if isinstance(layer, nn.Linear)
        for p in layer.parameters()
        if p.requires_grad
    )

    linear_indices = [idx for idx, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear)]
    if len(linear_indices) < 2:
        raise ValueError("Model classifier must expose at least two linear layers.")

    first_linear_idx, second_linear_idx = linear_indices[:2]
    first_linear = model.classifier[first_linear_idx]

    if not hasattr(model, "avgpool"):
        raise ValueError("Model does not expose an avgpool attribute.")

    output_size = getattr(model.avgpool, "output_size", None)
    if output_size is None:
        raise ValueError("Could not determine the output size of model.avgpool.")
    if isinstance(output_size, int):
        in_height, in_width = output_size, output_size
    else:
        in_height, in_width = output_size
    if in_height is None or in_width is None:
        raise ValueError("avgpool output size must resolve to concrete height and width.")

    in_features = first_linear.in_features
    expected_product = in_height * in_width
    if in_features % expected_product != 0:
        raise ValueError("First linear layer in_features is not divisible by avgpool spatial size.")
    in_channels = in_features // expected_product

    if debug:
        print("[replace_1st_2nd_fc_with_tcl] Classifier layout before modification:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")
        print(
            "[replace_1st_2nd_fc_with_tcl] Resolved first linear in_features="
            f"{in_features} (channels={in_channels}, height={in_height}, width={in_width})"
        )

    between_layers = list(model.classifier[first_linear_idx + 1 : second_linear_idx])
    tail_layers = list(model.classifier[second_linear_idx + 1 :])

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    second_flattened = second_out_channels * second_out_height * second_out_width

    # Adjust the first linear layer after the replacements (originally third linear / classifier[-1]).
    tail_linear_updated = False
    for idx, layer in enumerate(tail_layers):
        if isinstance(layer, nn.Linear):
            if debug:
                print(
                    "[replace_1st_2nd_fc_with_tcl] Adjusting remaining linear layer "
                    f"from in_features={layer.in_features} to {second_flattened}"
                )
            tail_layers[idx] = nn.Linear(
                second_flattened,
                layer.out_features,
                bias=layer.bias is not None,
            ).to(device=device, dtype=dtype)
            tail_linear_updated = True
            break
    if not tail_linear_updated:
        raise ValueError("Could not locate the final linear layer in the classifier to adjust.")

    bn1_in = nn.BatchNorm2d(in_channels).to(device=device, dtype=dtype)
    bn1_out = nn.BatchNorm2d(first_out_channels).to(device=device, dtype=dtype)
    tcl1 = TCL(
        in_channels=in_channels,
        out_channels=first_out_channels,
        out_height=first_out_height,
        out_width=first_out_width,
    ).to(device=device, dtype=dtype)

    bn2_in = nn.BatchNorm2d(first_out_channels).to(device=device, dtype=dtype)
    bn2_out = nn.BatchNorm2d(second_out_channels).to(device=device, dtype=dtype)
    tcl2 = TCL(
        in_channels=first_out_channels,
        out_channels=second_out_channels,
        out_height=second_out_height,
        out_width=second_out_width,
    ).to(device=device, dtype=dtype)

    new_classifier_layers: List[nn.Module] = [
        nn.Unflatten(1, (in_channels, in_height, in_width)),
        bn1_in,
        tcl1,
        bn1_out,
        *between_layers,
        bn2_in,
        tcl2,
        bn2_out,
        nn.Flatten(),
        *tail_layers,
    ]

    model.classifier = nn.Sequential(*new_classifier_layers)

    if debug:
        print("[replace_1st_2nd_fc_with_tcl] Classifier layout after modification:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    tracked_layers = tuple(
        layer for layer in model.classifier if isinstance(layer, (nn.Linear, TCL, QTCL))
    )
    modified_fc_params = sum(
        p.numel()
        for layer in tracked_layers
        for p in layer.parameters()
        if p.requires_grad
    )
    if modified_fc_params <= 0 or original_fc_params <= 0:
        space_saving = float("nan")
    else:
        space_saving = 1.0 - (modified_fc_params / original_fc_params)

    print(
        "[replace_1st_2nd_fc_with_tcl] Parameter comparison "
        f"| original_fc={original_fc_params:,} "
        f"| modified_fc={modified_fc_params:,} "
        f"| space_saving={space_saving:.4f}"
    )
    return model


def replace_1st_fc_with_qtcl(
    model: nn.Module,
    out_channels: int,
    out_height: int,
    out_width: int,
    *,
    use_batchnorm: bool = True,
    qmtl_kwargs: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> nn.Module:
    if not hasattr(model, "classifier"):
        raise ValueError("Model does not expose a classifier attribute.")
    if not isinstance(model.classifier, nn.Sequential) or len(model.classifier) == 0:
        raise ValueError("Model classifier must be a non-empty nn.Sequential.")

    original_fc_params = sum(
        p.numel()
        for layer in model.classifier
        if isinstance(layer, nn.Linear)
        for p in layer.parameters()
        if p.requires_grad
    )

    first_layer = model.classifier[0]
    if not isinstance(first_layer, nn.Linear):
        raise ValueError("The first classifier layer is not a linear layer.")

    if not hasattr(model, "avgpool"):
        raise ValueError("Model does not expose an avgpool attribute.")

    output_size = getattr(model.avgpool, "output_size", None)
    if output_size is None:
        raise ValueError("Could not determine the output size of model.avgpool.")
    if isinstance(output_size, int):
        in_height, in_width = output_size, output_size
    else:
        in_height, in_width = output_size

    if in_height is None or in_width is None:
        raise ValueError("avgpool output size must resolve to concrete height and width.")

    in_features = first_layer.in_features
    expected_product = in_height * in_width
    if in_features % expected_product != 0:
        raise ValueError("First linear layer in_features is not divisible by avgpool spatial size.")

    in_channels = in_features // expected_product
    flattened_features = out_channels * out_height * out_width

    rest_layers = list(model.classifier[1:])

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if debug:
        print("[replace_1st_fc_with_qtcl] Original classifier:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    updated_linear = False
    for idx, layer in enumerate(rest_layers):
        if isinstance(layer, nn.Linear):
            if debug:
                print(
                    "[replace_1st_fc_with_qtcl] Adjusting next linear layer "
                    f"from in_features={layer.in_features} to {flattened_features}"
                )
            rest_layers[idx] = nn.Linear(
                flattened_features,
                layer.out_features,
                bias=layer.bias is not None,
            ).to(device=device, dtype=dtype)
            updated_linear = True
            break
    if not updated_linear:
        raise ValueError("Could not locate the next linear layer in the classifier to adjust.")

    if use_batchnorm:
        bn_in: nn.Module = nn.BatchNorm2d(in_channels).to(device=device, dtype=dtype)
        bn_out: nn.Module = nn.BatchNorm2d(out_channels).to(device=device, dtype=dtype)
    else:
        bn_in = nn.Identity()
        bn_out = nn.Identity()

    q_kwargs: Dict[str, Any] = dict(qmtl_kwargs or {})
    for forbidden in ("in_channels", "out_channels", "out_height", "out_width"):
        q_kwargs.pop(forbidden, None)

    qtcl_layer = QTCL(
        in_channels=in_channels,
        out_channels=out_channels,
        out_height=out_height,
        out_width=out_width,
        **q_kwargs,
    ).to(device=device, dtype=dtype)

    new_classifier_layers = [
        nn.Unflatten(1, (in_channels, in_height, in_width)),
        bn_in,
        qtcl_layer,
        bn_out,
        nn.Flatten(),
        *rest_layers,
    ]
    model.classifier = nn.Sequential(*new_classifier_layers)

    if debug:
        print("[replace_1st_fc_with_qtcl] New classifier:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    tracked_layers = tuple(
        layer for layer in model.classifier if isinstance(layer, (nn.Linear, TCL, QTCL))
    )
    modified_fc_params = sum(
        p.numel()
        for layer in tracked_layers
        for p in layer.parameters()
        if p.requires_grad
    )
    if modified_fc_params <= 0 or original_fc_params <= 0:
        space_saving = float("nan")
    else:
        space_saving = 1.0 - (modified_fc_params / original_fc_params)

    print(
        "[replace_1st_fc_with_qtcl] Parameter comparison "
        f"| original_fc={original_fc_params:,} "
        f"| modified_fc={modified_fc_params:,} "
        f"| space_saving={space_saving:.4f}"
    )
    return model


def replace_1st_2nd_fc_with_qtcl(
    model: nn.Module,
    first_out_channels: int,
    first_out_height: int,
    first_out_width: int,
    second_out_channels: int,
    second_out_height: int,
    second_out_width: int,
    *,
    use_batchnorm: bool = True,
    first_qmtl_kwargs: Optional[Dict[str, Any]] = None,
    second_qmtl_kwargs: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> nn.Module:
    if not hasattr(model, "classifier"):
        raise ValueError("Model does not expose a classifier attribute.")
    if not isinstance(model.classifier, nn.Sequential) or len(model.classifier) == 0:
        raise ValueError("Model classifier must be a non-empty nn.Sequential.")

    original_fc_params = sum(
        p.numel()
        for layer in model.classifier
        if isinstance(layer, nn.Linear)
        for p in layer.parameters()
        if p.requires_grad
    )

    linear_indices = [idx for idx, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear)]
    if len(linear_indices) < 2:
        raise ValueError("Model classifier must expose at least two linear layers.")

    first_linear_idx, second_linear_idx = linear_indices[:2]
    first_linear = model.classifier[first_linear_idx]

    if not hasattr(model, "avgpool"):
        raise ValueError("Model does not expose an avgpool attribute.")

    output_size = getattr(model.avgpool, "output_size", None)
    if output_size is None:
        raise ValueError("Could not determine the output size of model.avgpool.")
    if isinstance(output_size, int):
        in_height, in_width = output_size, output_size
    else:
        in_height, in_width = output_size
    if in_height is None or in_width is None:
        raise ValueError("avgpool output size must resolve to concrete height and width.")

    in_features = first_linear.in_features
    expected_product = in_height * in_width
    if in_features % expected_product != 0:
        raise ValueError("First linear layer in_features is not divisible by avgpool spatial size.")
    in_channels = in_features // expected_product

    if debug:
        print("[replace_1st_2nd_fc_with_qtcl] Classifier layout before modification:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    between_layers = list(model.classifier[first_linear_idx + 1 : second_linear_idx])
    tail_layers = list(model.classifier[second_linear_idx + 1 :])

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    second_flattened = second_out_channels * second_out_height * second_out_width

    tail_linear_updated = False
    for idx, layer in enumerate(tail_layers):
        if isinstance(layer, nn.Linear):
            if debug:
                print(
                    "[replace_1st_2nd_fc_with_qtcl] Adjusting remaining linear layer "
                    f"from in_features={layer.in_features} to {second_flattened}"
                )
            tail_layers[idx] = nn.Linear(
                second_flattened,
                layer.out_features,
                bias=layer.bias is not None,
            ).to(device=device, dtype=dtype)
            tail_linear_updated = True
            break
    if not tail_linear_updated:
        raise ValueError("Could not locate the final linear layer in the classifier to adjust.")

    bn1_in = nn.BatchNorm2d(in_channels).to(device=device, dtype=dtype) if use_batchnorm else None
    bn1_out = nn.BatchNorm2d(first_out_channels).to(device=device, dtype=dtype) if use_batchnorm else None
    bn2_out = nn.BatchNorm2d(second_out_channels).to(device=device, dtype=dtype) if use_batchnorm else None

    first_kwargs: Dict[str, Any] = dict(first_qmtl_kwargs or {})
    second_kwargs: Dict[str, Any] = dict(second_qmtl_kwargs or {})
    for forbidden in ("in_channels", "out_channels", "out_height", "out_width"):
        first_kwargs.pop(forbidden, None)
        second_kwargs.pop(forbidden, None)

    qtcl1 = QTCL(
        in_channels=in_channels,
        out_channels=first_out_channels,
        out_height=first_out_height,
        out_width=first_out_width,
        **first_kwargs,
    ).to(device=device, dtype=dtype)

    qtcl2 = QTCL(
        in_channels=first_out_channels,
        out_channels=second_out_channels,
        out_height=second_out_height,
        out_width=second_out_width,
        **second_kwargs,
    ).to(device=device, dtype=dtype)

    new_classifier_layers: List[nn.Module] = [nn.Unflatten(1, (in_channels, in_height, in_width))]
    if bn1_in is not None:
        new_classifier_layers.append(bn1_in)
    new_classifier_layers.append(qtcl1)
    if bn1_out is not None:
        new_classifier_layers.append(bn1_out)
    new_classifier_layers.extend(between_layers)
    new_classifier_layers.append(qtcl2)
    if bn2_out is not None:
        new_classifier_layers.append(bn2_out)
    new_classifier_layers.append(nn.Flatten())
    new_classifier_layers.extend(tail_layers)

    model.classifier = nn.Sequential(*new_classifier_layers)

    if debug:
        print("[replace_1st_2nd_fc_with_qtcl] Classifier layout after modification:")
        for idx, layer in enumerate(model.classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    tracked_layers = tuple(
        layer for layer in model.classifier if isinstance(layer, (nn.Linear, TCL, QTCL))
    )
    modified_fc_params = sum(
        p.numel()
        for layer in tracked_layers
        for p in layer.parameters()
        if p.requires_grad
    )
    if modified_fc_params <= 0 or original_fc_params <= 0:
        space_saving = float("nan")
    else:
        space_saving = 1.0 - (modified_fc_params / original_fc_params)

    print(
        "[replace_1st_2nd_fc_with_qtcl] Parameter comparison "
        f"| original_fc={original_fc_params:,} "
        f"| modified_fc={modified_fc_params:,} "
        f"| space_saving={space_saving:.4f}"
    )
    return model


def build_model(num_classes: int, pretrained: bool, device: torch.device) -> nn.Module:
    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg19(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool2d((3, 3))
    model.classifier[0] = nn.Linear(512 * 3 * 3, 4096, bias=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes, bias=True)
    model.to(device)
    return model


def dummy_forward_pass(model: nn.Module, device: torch.device, log: bool = True) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        output = model(dummy_input)
    if log:
        print("Dummy output shape:", output.shape)
    model.train(was_training)


def _extract_model_state_dict(checkpoint_payload: Any, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["model_state_dict"]
    elif isinstance(checkpoint_payload, dict):
        state_dict = checkpoint_payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type in {checkpoint_path}: {type(checkpoint_payload).__name__}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a valid model state dict.")

    if any(str(key).startswith("module.") for key in state_dict.keys()):
        return {
            str(key)[len("module.") :] if str(key).startswith("module.") else str(key): value
            for key, value in state_dict.items()
        }
    return dict(state_dict)


def _format_key_preview(keys: List[str], limit: int = 6) -> str:
    if not keys:
        return "none"
    preview = ", ".join(keys[:limit])
    if len(keys) > limit:
        preview += f", ... (+{len(keys) - limit} more)"
    return preview


def initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    allow_partial: bool = False,
    log: bool = True,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {checkpoint_path}")

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_model_state_dict(checkpoint_payload, checkpoint_path)
    model_state = model.state_dict()

    if allow_partial:
        matched_state: Dict[str, torch.Tensor] = {}
        shape_mismatches: List[str] = []
        unexpected_keys: List[str] = []
        for key, value in state_dict.items():
            if key not in model_state:
                unexpected_keys.append(str(key))
                continue
            if model_state[key].shape != value.shape:
                shape_mismatches.append(
                    f"{key}: checkpoint{tuple(value.shape)} != model{tuple(model_state[key].shape)}"
                )
                continue
            matched_state[str(key)] = value

        if not matched_state:
            raise RuntimeError(
                f"Failed to partially initialize model from {checkpoint_path}: no compatible tensors were found."
            )

        missing_keys = [key for key in model_state.keys() if key not in matched_state]
        model.load_state_dict(matched_state, strict=False)
    else:
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load initialization checkpoint from {checkpoint_path}.") from exc
        matched_state = dict(state_dict)
        missing_keys = []
        shape_mismatches = []
        unexpected_keys = []

    if log:
        details = []
        if isinstance(checkpoint_payload, dict):
            if "epoch" in checkpoint_payload:
                details.append(f"epoch={checkpoint_payload['epoch']}")
            if "best_val_acc" in checkpoint_payload:
                details.append(f"best_val_acc={float(checkpoint_payload['best_val_acc']) * 100:.2f}%")
        detail_suffix = f" ({', '.join(details)})" if details else ""
        if allow_partial and (missing_keys or shape_mismatches or unexpected_keys):
            print(
                "Partially initialized model weights from checkpoint: "
                f"{checkpoint_path}{detail_suffix} "
                f"| loaded={len(matched_state)}/{len(model_state)} tensors "
                f"| missing={len(missing_keys)} "
                f"| shape_mismatches={len(shape_mismatches)} "
                f"| unexpected={len(unexpected_keys)}"
            )
            if missing_keys:
                print(f"  Missing keys: {_format_key_preview(missing_keys)}")
            if shape_mismatches:
                print(f"  Shape mismatches: {_format_key_preview(shape_mismatches)}")
            if unexpected_keys:
                print(f"  Unexpected keys: {_format_key_preview(unexpected_keys)}")
        else:
            print(f"Initialized model weights from checkpoint: {checkpoint_path}{detail_suffix}")


def get_final_classifier_linear(model: nn.Module) -> nn.Linear:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential):
        raise ValueError("Model does not expose a sequential classifier.")

    for layer in reversed(classifier):
        if isinstance(layer, nn.Linear):
            return layer
    raise ValueError("Could not locate the final classifier linear layer.")


def get_classifier_fc_like_layers(model: nn.Module) -> List[Tuple[int, nn.Module]]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential):
        raise ValueError("Model does not expose a sequential classifier.")
    return [
        (idx, layer)
        for idx, layer in enumerate(classifier)
        if isinstance(layer, (nn.Linear, TCL, QTCL))
    ]


def freeze_all_but_classifier(model: nn.Module, log: bool = True) -> None:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential):
        raise ValueError("Model does not expose a sequential classifier.")

    for param in model.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = True

    if log:
        trainable_params = sum(param.numel() for param in classifier.parameters())
        print(f"Training scope: classifier head only | trainable_params={trainable_params}")


def freeze_all_but_final_classifier(model: nn.Module, log: bool = True) -> None:
    final_linear = get_final_classifier_linear(model)
    for param in model.parameters():
        param.requires_grad = False
    for param in final_linear.parameters():
        param.requires_grad = True

    if log:
        trainable_params = sum(param.numel() for param in final_linear.parameters())
        print(
            "Training scope: final classifier layer only "
            f"| in_features={final_linear.in_features} | out_features={final_linear.out_features} "
            f"| trainable_params={trainable_params}"
        )


def freeze_all_but_last_classifier_fc_layers(model: nn.Module, num_layers: int, log: bool = True) -> None:
    if num_layers <= 0:
        raise ValueError("num_layers must be a positive integer.")

    fc_like_layers = get_classifier_fc_like_layers(model)
    if len(fc_like_layers) < num_layers:
        raise ValueError(
            f"Classifier exposes only {len(fc_like_layers)} FC-like layers, cannot keep the last {num_layers} trainable."
        )

    for param in model.parameters():
        param.requires_grad = False

    selected_layers = fc_like_layers[-num_layers:]
    for _, layer in selected_layers:
        for param in layer.parameters():
            param.requires_grad = True

    if log:
        layer_summary = ", ".join(f"[{idx}] {layer.__class__.__name__}" for idx, layer in selected_layers)
        trainable_params = sum(param.numel() for _, layer in selected_layers for param in layer.parameters())
        print(
            f"Training scope: last {num_layers} classifier FC-like layers only "
            f"| layers={layer_summary} | trainable_params={trainable_params}"
        )


def set_frozen_batchnorm_modules_eval(model: nn.Module) -> None:
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if any(param.requires_grad for param in module.parameters(recurse=False)):
            continue
        module.eval()


def collect_qtcl_modules(model: nn.Module) -> List[QTCL]:
    return [module for module in model.modules() if isinstance(module, QTCL)]


def split_qtcl_parameter_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    qtcl_modules = collect_qtcl_modules(model)
    if not qtcl_modules:
        backbone_params = [p for p in model.parameters() if p.requires_grad]
        return backbone_params, []

    quantum_params: List[nn.Parameter] = []
    seen: Set[int] = set()
    for module in qtcl_modules:
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                quantum_params.append(param)
                seen.add(id(param))

    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    return backbone_params, quantum_params


def _quantum_block_param_stats(module: QTCL) -> Dict[str, Tuple[int, float]]:
    def _stats(parameters: Iterable[nn.Parameter]) -> Tuple[int, float]:
        total_params = 0
        total_sum = 0.0
        for param in parameters:
            if not param.requires_grad:
                continue
            data = param.detach()
            total_params += data.numel()
            total_sum += data.sum().item()
        mean_val = total_sum / total_params if total_params > 0 else float("nan")
        return total_params, mean_val

    stats = {
        "B": _stats(module.B.parameters()),
        "PQC": _stats(module.q_layer.parameters()),
        "A": _stats(module.A.parameters()),
    }
    return stats


def log_qtcl_parameter_debug(
    model: nn.Module,
    epoch: int,
    main_process: bool,
) -> None:
    if not main_process:
        return
    param_model = model.module if isinstance(model, DDP) else model
    qtcl_modules = collect_qtcl_modules(param_model)
    if not qtcl_modules:
        return
    print(f"[QTCL Debug] Epoch {epoch:02d}")
    for idx, module in enumerate(qtcl_modules, start=1):
        stats = _quantum_block_param_stats(module)
        alpha_repr = module._alpha_repr()
        classical_mean = getattr(module, "_last_classical_mean", None)
        quantum_mean = getattr(module, "_last_quantum_mean", None)
        classical_str = f"{classical_mean:+.5e}" if classical_mean is not None else "n/a"
        quantum_str = f"{quantum_mean:+.5e}" if quantum_mean is not None else "n/a"
        print(
            f"  Block {idx} | alpha={alpha_repr} "
            f"| B(mean={stats['B'][1]:+.5e}, params={stats['B'][0]:,}) "
            f"| PQC(mean={stats['PQC'][1]:+.5e}, params={stats['PQC'][0]:,}) "
            f"| A(mean={stats['A'][1]:+.5e}, params={stats['A'][0]:,}) "
            f"| activations(classical={classical_str}, quantum={quantum_str})"
        )


def _deterministic_subset_indices(dataset_size: int, seed: Optional[int]) -> List[int]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    # A fixed seed always produces the same ordering, so smaller fractions are
    # strict prefixes of larger fractions. For example, 10% is guaranteed to be
    # a subset of 20%, 20% of 40%, and so on for the same dataset and seed.
    return torch.randperm(dataset_size, generator=generator).tolist()


def _subset_dataset(dataset, fraction: float, seed: Optional[int]):
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError("fraction must be in the interval (0, 1].")

    subset_size = max(1, math.ceil(fraction * len(dataset)))
    indices = _deterministic_subset_indices(len(dataset), seed)[:subset_size]
    return Subset(dataset, indices)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    workers: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Tuple[
    DataLoader,
    DataLoader,
    Optional[DistributedSampler],
    Optional[DistributedSampler],
]:
    normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)

    train_dataset = _subset_dataset(train_dataset, train_fraction, seed)
    val_dataset = _subset_dataset(test_dataset, val_fraction, seed + 1 if seed is not None else None)

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
            seed=seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=seed + 1 if seed is not None else 0,
        )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1 if seed is not None else 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=val_generator,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    quantum_optimizer: Optional[optim.Optimizer],
    freeze_backbone: bool,
    train_final_layer_only: bool,
    train_classifier_only: bool,
    train_last_classifier_fc_layers: Optional[int],
    device: torch.device,
    epoch_desc: str,
    show_progress: bool,
    distributed: bool,
) -> Tuple[float, float, bool]:
    freeze_modes = int(train_final_layer_only) + int(train_classifier_only) + int(train_last_classifier_fc_layers is not None)
    if freeze_modes > 1:
        raise ValueError("Specify at most one selective training scope.")

    if train_final_layer_only:
        # Keep the frozen feature extractor deterministic by disabling training
        # behavior such as dropout and BatchNorm running-stat updates.
        model.eval()
    else:
        model.train()
        if train_last_classifier_fc_layers is not None:
            set_frozen_batchnorm_modules_eval(model)
    running_loss = 0.0
    running_correct = 0
    total = 0
    backbone_stepped = False
    iterator = tqdm(dataloader, desc=epoch_desc, leave=False, disable=not show_progress, mininterval=1.0)
    total_steps = len(dataloader)
    for step_idx, (images, targets) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if quantum_optimizer is not None:
            quantum_optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        if not freeze_backbone:
            optimizer.step()
            backbone_stepped = True
        if quantum_optimizer is not None:
            quantum_optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        running_correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if show_progress and total > 0 and (step_idx == 1 or step_idx % 10 == 0 or step_idx == total_steps):
            iterator.set_postfix(
                loss=running_loss / total,
                acc=running_correct / total,
            )

    stats = torch.tensor([running_loss, running_correct, total], device=device, dtype=torch.float64)
    if distributed and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    running_loss = stats[0].item()
    running_correct = stats[1].item()
    total = int(stats[2].item())
    total = max(total, 1)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc, backbone_stepped


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_desc: str,
    show_progress: bool,
    distributed: bool,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    iterator = tqdm(dataloader, desc=epoch_desc, leave=False, disable=not show_progress, mininterval=1.0)
    total_steps = len(dataloader)
    for step_idx, (images, targets) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        running_correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if show_progress and total > 0 and (step_idx == 1 or step_idx % 10 == 0 or step_idx == total_steps):
            iterator.set_postfix(
                loss=running_loss / total,
                acc=running_correct / total,
            )

    stats = torch.tensor([running_loss, running_correct, total], device=device, dtype=torch.float64)
    if distributed and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    running_loss = stats[0].item()
    running_correct = stats[1].item()
    total = int(stats[2].item())
    total = max(total, 1)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def init_distributed(args: argparse.Namespace) -> torch.device:
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = args.world_size > 1

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA availability.")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def cleanup_distributed(args: argparse.Namespace) -> None:
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args: argparse.Namespace) -> bool:
    return getattr(args, "rank", 0) == 0


def format_device_for_log(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    if device.index is not None:
        return f"cuda:{device.index}"
    try:
        return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        return "cuda"


def serialize_for_storage(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_for_storage(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_storage(item) for item in value]
    return value


def resolve_run_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds is not None:
        resolved = [int(seed) for seed in args.seeds]
    else:
        if args.num_seeds <= 0:
            raise ValueError("num-seeds must be a positive integer.")
        resolved = [args.seed + offset for offset in range(args.num_seeds)]

    if not resolved:
        raise ValueError("Resolved seed list is empty.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Resolved seed list contains duplicates.")
    return resolved


def resolve_run_output_dir(root_output_dir: Path, run_seed: int, multi_seed: bool) -> Path:
    return root_output_dir / f"seed_{run_seed}" if multi_seed else root_output_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean_value = sum(values) / len(values)
    if len(values) == 1:
        return mean_value, 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return mean_value, math.sqrt(variance)


def save_single_run_artifacts(
    output_dir: Path,
    train_losses: List[float],
    train_accuracies: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    best_val_acc: float,
    best_val_acc_epoch: int,
    best_val_loss: float,
    best_val_loss_epoch: int,
) -> Tuple[Dict[str, float], Path, Path, Path]:
    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch_idx, (tr_loss, tr_acc, va_loss, va_acc) in enumerate(
            zip(train_losses, train_accuracies, val_losses, val_accuracies),
            start=1,
        ):
            writer.writerow([epoch_idx, tr_loss, tr_acc, va_loss, va_acc])

    summary = {
        "best_val_acc": best_val_acc,
        "best_val_acc_epoch": float(best_val_acc_epoch),
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": float(best_val_loss_epoch),
        "final_train_loss": train_losses[-1],
        "final_train_acc": train_accuracies[-1],
        "final_val_loss": val_losses[-1],
        "final_val_acc": val_accuracies[-1],
    }

    summary_path = output_dir / "training_summary.csv"
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "best_val_acc",
                "best_val_acc_epoch",
                "best_val_loss",
                "best_val_loss_epoch",
                "final_train_loss",
                "final_train_acc",
                "final_val_loss",
                "final_val_acc",
            ]
        )
        writer.writerow(
            [
                summary["best_val_acc"],
                int(summary["best_val_acc_epoch"]),
                summary["best_val_loss"],
                int(summary["best_val_loss_epoch"]),
                summary["final_train_loss"],
                summary["final_train_acc"],
                summary["final_val_loss"],
                summary["final_val_acc"],
            ]
        )

    figure_path = output_dir / "training_curves.png"
    epochs_range = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs_range, train_losses, label="Train")
    axes[0].plot(epochs_range, val_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()

    axes[1].plot(epochs_range, [acc * 100 for acc in train_accuracies], label="Train")
    axes[1].plot(epochs_range, [acc * 100 for acc in val_accuracies], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return summary, log_path, summary_path, figure_path


def write_seed_summaries(output_dir: Path, run_records: List[Dict[str, Any]]) -> Path:
    seed_summaries_path = output_dir / "seed_summaries.csv"
    with seed_summaries_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "seed",
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
        )
        for record in run_records:
            writer.writerow(
                [
                    record["seed"],
                    record["run_output_dir"],
                    record["best_val_acc"],
                    record["best_val_acc_epoch"],
                    record["best_val_loss"],
                    record["best_val_loss_epoch"],
                    record["final_train_loss"],
                    record["final_train_acc"],
                    record["final_val_loss"],
                    record["final_val_acc"],
                ]
            )
    return seed_summaries_path


def write_aggregate_summary(output_dir: Path, run_records: List[Dict[str, Any]]) -> Path:
    aggregate_summary_path = output_dir / "aggregate_summary.csv"
    fields = [
        "best_val_acc",
        "best_val_acc_epoch",
        "best_val_loss",
        "best_val_loss_epoch",
        "final_train_loss",
        "final_train_acc",
        "final_val_loss",
        "final_val_acc",
    ]

    aggregate_row: Dict[str, Any] = {
        "num_seeds": len(run_records),
        "seeds": " ".join(str(record["seed"]) for record in run_records),
    }
    for field in fields:
        values = [float(record[field]) for record in run_records]
        mean_value, std_value = compute_mean_std(values)
        aggregate_row[f"{field}_mean"] = mean_value
        aggregate_row[f"{field}_std"] = std_value

    with aggregate_summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(aggregate_row.keys()))
        writer.writeheader()
        writer.writerow(aggregate_row)
    return aggregate_summary_path


def write_aggregate_training_log(output_dir: Path, run_records: List[Dict[str, Any]]) -> Path:
    aggregate_log_path = output_dir / "aggregate_training_log.csv"
    histories = [record["history"] for record in run_records]
    num_epochs = len(histories[0]["train_losses"])
    for history in histories[1:]:
        if len(history["train_losses"]) != num_epochs:
            raise ValueError("All seed runs must complete the same number of epochs for aggregation.")

    with aggregate_log_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch",
                "num_seeds",
                "train_loss_mean",
                "train_loss_std",
                "train_acc_mean",
                "train_acc_std",
                "val_loss_mean",
                "val_loss_std",
                "val_acc_mean",
                "val_acc_std",
            ]
        )
        for epoch_idx in range(num_epochs):
            train_losses = [history["train_losses"][epoch_idx] for history in histories]
            train_accs = [history["train_accuracies"][epoch_idx] for history in histories]
            val_losses = [history["val_losses"][epoch_idx] for history in histories]
            val_accs = [history["val_accuracies"][epoch_idx] for history in histories]
            train_loss_mean, train_loss_std = compute_mean_std(train_losses)
            train_acc_mean, train_acc_std = compute_mean_std(train_accs)
            val_loss_mean, val_loss_std = compute_mean_std(val_losses)
            val_acc_mean, val_acc_std = compute_mean_std(val_accs)
            writer.writerow(
                [
                    epoch_idx + 1,
                    len(run_records),
                    train_loss_mean,
                    train_loss_std,
                    train_acc_mean,
                    train_acc_std,
                    val_loss_mean,
                    val_loss_std,
                    val_acc_mean,
                    val_acc_std,
                ]
            )
    return aggregate_log_path


def write_experiment_config(output_dir: Path, args: argparse.Namespace, resolved_seeds: List[int]) -> Path:
    config_path = output_dir / "experiment_config.json"
    payload = {
        "resolved_seeds": list(resolved_seeds),
        "num_seeds": len(resolved_seeds),
        "multi_seed": len(resolved_seeds) > 1,
        "args": serialize_for_storage(vars(args)),
    }
    write_json(config_path, payload)
    return config_path


def run_single_seed(
    args: argparse.Namespace,
    run_seed: int,
    run_index: int,
    total_runs: int,
    device: torch.device,
    main_process: bool,
    resolved_seeds: List[int],
) -> Dict[str, Any]:
    run_output_dir = resolve_run_output_dir(args.output_dir, run_seed, total_runs > 1)
    if main_process:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        label = f"Seed {run_seed} ({run_index}/{total_runs})" if total_runs > 1 else f"Seed {run_seed}"
        print(f"Starting {label} -> {run_output_dir}")

    seed_everything(run_seed)

    run_args = copy.deepcopy(args)
    run_args.seed = run_seed
    run_args.output_dir = run_output_dir
    run_args.resolved_seeds = list(resolved_seeds)
    run_args.seed_run_index = run_index
    run_args.seed_run_count = total_runs
    run_args.aggregate_output_dir = args.output_dir

    model = build_model(num_classes=100, pretrained=run_args.pretrained, device=device)

    model_variant = run_args.model
    if model_variant == "vgg19_tcl":
        model_variant = "vgg19_tcl1"
    if model_variant == "vgg19_qtcl":
        model_variant = "vgg19_qtcl1"

    first_stage_variants = {"vgg19_tcl1", "vgg19_tcl12", "vgg19_qtcl1", "vgg19_qtcl12"}
    second_stage_variants = {"vgg19_tcl12", "vgg19_qtcl12"}

    if model_variant in first_stage_variants:
        if (
            run_args.tcl1_out_channels <= 0
            or run_args.tcl1_out_height <= 0
            or run_args.tcl1_out_width <= 0
        ):
            raise ValueError("First replacement output dimensions must be positive integers.")

    if model_variant in second_stage_variants:
        if (
            run_args.tcl2_out_channels <= 0
            or run_args.tcl2_out_height <= 0
            or run_args.tcl2_out_width <= 0
        ):
            raise ValueError("Second replacement output dimensions must be positive integers.")

    qtcl_first_kwargs: Optional[Dict[str, Any]] = None
    qtcl_second_kwargs: Optional[Dict[str, Any]] = None
    if model_variant in {"vgg19_qtcl1", "vgg19_qtcl12"}:
        if run_args.qtcl_n_qubits <= 0:
            raise ValueError("qtcl-n-qubits must be a positive integer.")
        if run_args.qtcl_n_layers <= 0:
            raise ValueError("qtcl-n-layers must be a positive integer.")
        if run_args.qtcl_shots < 0:
            raise ValueError("qtcl-shots must be non-negative (0 means analytic expectation).")
        if run_args.qtcl_F is not None and run_args.qtcl_F <= 0:
            raise ValueError("qtcl-F must be a positive integer when provided.")
        if not 0.0 <= run_args.qtcl_alpha <= 1.0:
            raise ValueError("qtcl-alpha must lie in the interval [0, 1].")

        try:
            qtcl_ansatz_kwargs_loaded = json.loads(run_args.qtcl_ansatz_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse --qtcl-ansatz-kwargs as JSON.") from exc
        if not isinstance(qtcl_ansatz_kwargs_loaded, dict):
            raise ValueError("--qtcl-ansatz-kwargs must decode to a JSON object.")

        qtcl_first_kwargs = {
            "n_qubits": run_args.qtcl_n_qubits,
            "n_layers": run_args.qtcl_n_layers,
            "F": run_args.qtcl_F,
            "shots": None if run_args.qtcl_shots == 0 else run_args.qtcl_shots,
            "ansatz_type": run_args.qtcl_ansatz,
            "ansatz_kwargs": dict(qtcl_ansatz_kwargs_loaded),
            "alpha": run_args.qtcl_alpha,
            "learnable_alpha": run_args.qtcl_learnable_alpha,
        }
        qtcl_second_kwargs = copy.deepcopy(qtcl_first_kwargs)

        if run_args.qtcl2_n_qubits is not None:
            if run_args.qtcl2_n_qubits <= 0:
                raise ValueError("qtcl2-n-qubits must be a positive integer when provided.")
            qtcl_second_kwargs["n_qubits"] = run_args.qtcl2_n_qubits
        if run_args.qtcl2_n_layers is not None:
            if run_args.qtcl2_n_layers <= 0:
                raise ValueError("qtcl2-n-layers must be a positive integer when provided.")
            qtcl_second_kwargs["n_layers"] = run_args.qtcl2_n_layers
        if run_args.qtcl2_F is not None:
            if run_args.qtcl2_F <= 0:
                raise ValueError("qtcl2-F must be a positive integer when provided.")
            qtcl_second_kwargs["F"] = run_args.qtcl2_F
        if run_args.qtcl2_shots is not None:
            if run_args.qtcl2_shots < 0:
                raise ValueError("qtcl2-shots must be non-negative (0 means analytic expectation).")
            qtcl_second_kwargs["shots"] = None if run_args.qtcl2_shots == 0 else run_args.qtcl2_shots
        if run_args.qtcl2_ansatz is not None:
            qtcl_second_kwargs["ansatz_type"] = run_args.qtcl2_ansatz
        if run_args.qtcl2_ansatz_kwargs is not None:
            try:
                qtcl2_ansatz_kwargs_loaded = json.loads(run_args.qtcl2_ansatz_kwargs)
            except json.JSONDecodeError as exc:
                raise ValueError("Failed to parse --qtcl2-ansatz-kwargs as JSON.") from exc
            if not isinstance(qtcl2_ansatz_kwargs_loaded, dict):
                raise ValueError("--qtcl2-ansatz-kwargs must decode to a JSON object.")
            qtcl_second_kwargs["ansatz_kwargs"] = dict(qtcl2_ansatz_kwargs_loaded)
        if run_args.qtcl2_alpha is not None:
            if not 0.0 <= run_args.qtcl2_alpha <= 1.0:
                raise ValueError("qtcl2-alpha must lie in the interval [0, 1].")
            qtcl_second_kwargs["alpha"] = run_args.qtcl2_alpha
        if run_args.qtcl2_learnable_alpha is not None:
            qtcl_second_kwargs["learnable_alpha"] = run_args.qtcl2_learnable_alpha

    if model_variant == "vgg19_tcl1":
        model = replace_1st_fc_with_tcl(
            model,
            out_channels=run_args.tcl1_out_channels,
            out_height=run_args.tcl1_out_height,
            out_width=run_args.tcl1_out_width,
            debug=run_args.tcl_debug and main_process,
        )
    elif model_variant == "vgg19_tcl12":
        model = replace_1st_2nd_fc_with_tcl(
            model,
            first_out_channels=run_args.tcl1_out_channels,
            first_out_height=run_args.tcl1_out_height,
            first_out_width=run_args.tcl1_out_width,
            second_out_channels=run_args.tcl2_out_channels,
            second_out_height=run_args.tcl2_out_height,
            second_out_width=run_args.tcl2_out_width,
            debug=run_args.tcl_debug and main_process,
        )
    elif model_variant == "vgg19_qtcl1":
        model = replace_1st_fc_with_qtcl(
            model,
            out_channels=run_args.tcl1_out_channels,
            out_height=run_args.tcl1_out_height,
            out_width=run_args.tcl1_out_width,
            use_batchnorm=True,
            qmtl_kwargs=qtcl_first_kwargs,
            debug=run_args.tcl_debug and main_process,
        )
    elif model_variant == "vgg19_qtcl12":
        model = replace_1st_2nd_fc_with_qtcl(
            model,
            first_out_channels=run_args.tcl1_out_channels,
            first_out_height=run_args.tcl1_out_height,
            first_out_width=run_args.tcl1_out_width,
            second_out_channels=run_args.tcl2_out_channels,
            second_out_height=run_args.tcl2_out_height,
            second_out_width=run_args.tcl2_out_width,
            use_batchnorm=run_args.qtcl_batchnorm,
            first_qmtl_kwargs=qtcl_first_kwargs,
            second_qmtl_kwargs=qtcl_second_kwargs,
            debug=run_args.tcl_debug and main_process,
        )

    if run_args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    dummy_forward_pass(model, device, log=main_process)
    if run_args.init_checkpoint is not None:
        initialize_model_from_checkpoint(
            model,
            run_args.init_checkpoint,
            allow_partial=run_args.allow_partial_init,
            log=main_process,
        )
    if run_args.train_classifier_only:
        freeze_all_but_classifier(model, log=main_process)
    elif run_args.train_final_layer_only:
        freeze_all_but_final_classifier(model, log=main_process)
    elif run_args.train_last_classifier_fc_layers is not None:
        freeze_all_but_last_classifier_fc_layers(
            model,
            run_args.train_last_classifier_fc_layers,
            log=main_process,
        )

    if run_args.distributed:
        model = DDP(model, device_ids=[run_args.local_rank], output_device=run_args.local_rank, broadcast_buffers=False)

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        run_args.data_dir,
        run_args.batch_size,
        run_args.num_workers,
        run_args.train_fraction,
        run_args.val_fraction,
        run_args.seed,
        run_args.distributed,
        run_args.rank,
        run_args.world_size,
    )
    if main_process:
        print(f"Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")

    param_model = model.module if isinstance(model, DDP) else model
    backbone_params, quantum_params = split_qtcl_parameter_groups(param_model)
    if not backbone_params:
        raise ValueError("Backbone parameter group is empty; expected at least one non-quantum parameter.")

    criterion = nn.CrossEntropyLoss()
    if run_args.optimizer == "sgd":
        optimizer = optim.SGD(
            backbone_params,
            lr=run_args.lr,
            momentum=run_args.momentum,
            weight_decay=run_args.weight_decay,
            nesterov=False,
        )
    elif run_args.optimizer == "adam":
        optimizer = optim.Adam(
            backbone_params,
            lr=run_args.lr,
            betas=(0.9, 0.999),
            weight_decay=run_args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {run_args.optimizer}")
    quantum_optimizer: Optional[optim.Optimizer] = None
    if quantum_params:
        if run_args.qtcl_optimizer == "adam":
            quantum_optimizer = optim.Adam(
                quantum_params,
                lr=run_args.qtcl_lr,
                betas=run_args.qtcl_betas,
                weight_decay=run_args.qtcl_weight_decay,
            )
        elif run_args.qtcl_optimizer == "adamw":
            quantum_optimizer = optim.AdamW(
                quantum_params,
                lr=run_args.qtcl_lr,
                betas=run_args.qtcl_betas,
                weight_decay=run_args.qtcl_weight_decay,
            )
        else:
            raise ValueError(f"Unsupported qtcl optimizer: {run_args.qtcl_optimizer}")
    if run_args.lr_step_size <= 0:
        raise ValueError("lr_step_size must be a positive integer.")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=run_args.lr_step_size, gamma=run_args.lr_gamma)

    trainable_params = sum(p.numel() for p in param_model.parameters() if p.requires_grad)
    if main_process:
        print(f"Number of trainable parameters: {trainable_params}")

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    best_val_acc = 0.0
    best_val_acc_epoch = 0
    best_checkpoint_path: Optional[Path] = None

    epoch_prefix = f"Seed {run_seed} " if total_runs > 1 else ""
    for epoch in range(1, run_args.epochs + 1):
        if run_args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if run_args.distributed and val_sampler is not None:
            val_sampler.set_epoch(epoch)

        show_progress = (not run_args.no_progress) and main_process
        freeze_backbone = epoch <= run_args.freeze_backbone_epochs

        train_loss, train_acc, backbone_stepped = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            quantum_optimizer,
            freeze_backbone,
            run_args.train_final_layer_only,
            run_args.train_classifier_only,
            run_args.train_last_classifier_fc_layers,
            device,
            epoch_desc=f"{epoch_prefix}Epoch {epoch:02d} train",
            show_progress=show_progress,
            distributed=run_args.distributed,
        )
        log_qtcl_parameter_debug(model, epoch, main_process)
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch_desc=f"{epoch_prefix}Epoch {epoch:02d} val",
            show_progress=show_progress,
            distributed=run_args.distributed,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch

        previous_best = best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch

        if main_process and best_val_acc > previous_best:
            best_checkpoint_path = run_output_dir / "checkpoint_best_acc.pth"
            checkpoint_args = {k: serialize_for_storage(v) for k, v in vars(run_args).items()}
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": param_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "quantum_optimizer_state_dict": (
                    quantum_optimizer.state_dict() if quantum_optimizer is not None else None
                ),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "history": {
                    "train_losses": list(train_losses),
                    "val_losses": list(val_losses),
                    "train_accuracies": list(train_accuracies),
                    "val_accuracies": list(val_accuracies),
                },
                "args": checkpoint_args,
            }
            torch.save(checkpoint, best_checkpoint_path)

        if main_process:
            best_marker = " (new best)" if best_val_acc > previous_best else ""
            print(
                f"{epoch_prefix}Epoch {epoch:02d}/{run_args.epochs} "
                f"| train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}% "
                f"| val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}% "
                f"| best_acc={best_val_acc * 100:.2f}%{best_marker} "
                f"| lr={optimizer.param_groups[0]['lr']:.4e}"
            )

        if backbone_stepped:
            scheduler.step()
        elif epoch <= run_args.freeze_backbone_epochs and main_process:
            warnings.warn(
                "Skipped learning rate scheduler step because the backbone optimizer did not update; "
                "this is expected while --freeze-backbone-epochs is in effect.",
                RuntimeWarning,
            )

    summary = {
        "best_val_acc": best_val_acc,
        "best_val_acc_epoch": best_val_acc_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "final_train_loss": train_losses[-1],
        "final_train_acc": train_accuracies[-1],
        "final_val_loss": val_losses[-1],
        "final_val_acc": val_accuracies[-1],
    }
    figure_path: Optional[Path] = None
    if main_process:
        summary, _, _, figure_path = save_single_run_artifacts(
            run_output_dir,
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
            best_val_acc,
            best_val_acc_epoch,
            best_val_loss,
            best_val_loss_epoch,
        )

    if main_process:
        print(
            f"Seed {run_seed} summary "
            f"| best_val_acc={summary['best_val_acc'] * 100:.2f}% "
            f"| best_val_loss={summary['best_val_loss']:.4f}"
        )
        if best_checkpoint_path is not None:
            print(f"Best checkpoint saved to: {best_checkpoint_path}")
        else:
            print("Best checkpoint was not saved (no improvement).")
        if figure_path is not None:
            print(f"Training curves saved to: {figure_path}")

    return {
        "seed": run_seed,
        "run_output_dir": str(run_output_dir),
        "best_val_acc": summary["best_val_acc"],
        "best_val_acc_epoch": int(summary["best_val_acc_epoch"]),
        "best_val_loss": summary["best_val_loss"],
        "best_val_loss_epoch": int(summary["best_val_loss_epoch"]),
        "final_train_loss": summary["final_train_loss"],
        "final_train_acc": summary["final_train_acc"],
        "final_val_loss": summary["final_val_loss"],
        "final_val_acc": summary["final_val_acc"],
        "history": {
            "train_losses": list(train_losses),
            "train_accuracies": list(train_accuracies),
            "val_losses": list(val_losses),
            "val_accuracies": list(val_accuracies),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VGG19 on CIFAR-100 using PyTorch.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate for SGD optimizer.")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer used for non-quantum trainable parameters.",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for SGD optimizer.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose model weights are loaded before training for fine-tuning.",
    )
    parser.add_argument(
        "--allow-partial-init",
        action="store_true",
        help="Allow init-checkpoint loading to skip incompatible tensors when the architecture differs.",
    )
    parser.add_argument(
        "--train-final-layer-only",
        action="store_true",
        help="Freeze all parameters except the final classifier linear layer.",
    )
    parser.add_argument(
        "--train-classifier-only",
        action="store_true",
        help="Freeze all parameters except the classifier head.",
    )
    parser.add_argument(
        "--train-last-classifier-fc-layers",
        type=int,
        default=None,
        help="Freeze all parameters except the last N classifier FC-like layers (nn.Linear/TCL/QTCL).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vgg19",
        choices=[
            "vgg19",
            "vgg19_tcl",
            "vgg19_tcl1",
            "vgg19_tcl12",
            "vgg19_qtcl",
            "vgg19_qtcl1",
            "vgg19_qtcl12",
        ],
        help="Model variant to train.",
    )
    parser.add_argument(
        "--tcl-out-channels",
        "--tcl1-out-channels",
        dest="tcl1_out_channels",
        type=int,
        default=64,
        help=(
            "Output channel count for first replacement when using "
            "--model vgg19_tcl1/vgg19_tcl12/vgg19_qtcl1/vgg19_qtcl12."
        ),
    )
    parser.add_argument(
        "--tcl-out-height",
        "--tcl1-out-height",
        dest="tcl1_out_height",
        type=int,
        default=8,
        help=(
            "Output height for first replacement when using "
            "--model vgg19_tcl1/vgg19_tcl12/vgg19_qtcl1/vgg19_qtcl12."
        ),
    )
    parser.add_argument(
        "--tcl-out-width",
        "--tcl1-out-width",
        dest="tcl1_out_width",
        type=int,
        default=8,
        help=(
            "Output width for first replacement when using "
            "--model vgg19_tcl1/vgg19_tcl12/vgg19_qtcl1/vgg19_qtcl12."
        ),
    )
    parser.add_argument(
        "--tcl2-out-channels",
        type=int,
        default=32,
        help="Output channel count for second replacement when using --model vgg19_tcl12/vgg19_qtcl12.",
    )
    parser.add_argument(
        "--tcl2-out-height",
        type=int,
        default=8,
        help="Output height for second replacement when using --model vgg19_tcl12/vgg19_qtcl12.",
    )
    parser.add_argument(
        "--tcl2-out-width",
        type=int,
        default=8,
        help="Output width for second replacement when using --model vgg19_tcl12/vgg19_qtcl12.",
    )
    parser.add_argument("--tcl-debug", action="store_true", help="Enable verbose debugging for TCL/QTCL replacements.")
    parser.add_argument("--qtcl-n-qubits", type=int, default=8, help="Number of qubits for quantum TCL replacements.")
    parser.add_argument(
        "--qtcl-n-layers",
        type=int,
        default=2,
        help="Number of variational circuit layers for quantum TCL replacements.",
    )
    parser.add_argument(
        "--qtcl-F",
        type=int,
        default=None,
        help="Quantum latent feature width F for quantum TCL replacements (defaults to --qtcl-n-qubits).",
    )
    parser.add_argument(
        "--qtcl-alpha",
        type=float,
        default=0.5,
        help="Mixing coefficient between classical and quantum paths for the first QTCL block (0→classical, 1→quantum).",
    )
    parser.add_argument(
        "--qtcl-learnable-alpha",
        dest="qtcl_learnable_alpha",
        action="store_true",
        help="Make the first QTCL block mix coefficient alpha learnable.",
    )
    parser.add_argument(
        "--qtcl-fixed-alpha",
        dest="qtcl_learnable_alpha",
        action="store_false",
        help="Keep the first QTCL block mix coefficient alpha fixed (default).",
    )
    parser.set_defaults(qtcl_learnable_alpha=False)
    parser.add_argument(
        "--qtcl-shots",
        type=int,
        default=0,
        help="Quantum shots for TCL replacements (0 means analytic expectation).",
    )
    parser.add_argument(
        "--qtcl-ansatz",
        type=str,
        default="HEA",
        help="Variational ansatz key for quantum TCL replacements.",
    )
    parser.add_argument(
        "--qtcl-ansatz-kwargs",
        type=str,
        default="{}",
        help="JSON dict of extra kwargs forwarded to the quantum ansatz builder.",
    )
    parser.add_argument(
        "--qtcl2-n-qubits",
        type=int,
        default=None,
        help="Override qubit count for the second quantum TCL stage when using --model vgg19_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-n-layers",
        type=int,
        default=None,
        help="Override layer count for the second quantum TCL stage when using --model vgg19_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-F",
        type=int,
        default=None,
        help="Override latent feature width F for the second quantum TCL stage when using --model vgg19_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-alpha",
        type=float,
        default=None,
        help="Override the mixing coefficient alpha for the second QTCL block when using --model vgg19_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-learnable-alpha",
        dest="qtcl2_learnable_alpha",
        action="store_true",
        help="Override to make the second QTCL block mixing coefficient alpha learnable.",
    )
    parser.add_argument(
        "--qtcl2-fixed-alpha",
        dest="qtcl2_learnable_alpha",
        action="store_false",
        help="Override to keep the second QTCL block mixing coefficient alpha fixed.",
    )
    parser.set_defaults(qtcl2_learnable_alpha=None)
    parser.add_argument(
        "--qtcl2-shots",
        type=int,
        default=None,
        help="Override shots for the second quantum TCL stage when using --model vgg19_qtcl12 (0 means analytic).",
    )
    parser.add_argument(
        "--qtcl2-ansatz",
        type=str,
        default=None,
        help="Override ansatz key for the second quantum TCL stage when using --model vgg19_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-ansatz-kwargs",
        type=str,
        default=None,
        help="Override ansatz kwargs (JSON dict) for the second quantum TCL stage when using --model vgg19_qtcl12.",
    )
    parser.add_argument(
        "--qtcl-lr",
        type=float,
        default=5e-3,
        help="Learning rate for quantum TCL parameters when a dedicated optimizer is used.",
    )
    parser.add_argument(
        "--qtcl-optimizer",
        type=str,
        choices=["adamw", "adam"],
        default="adamw",
        help="Optimizer used for trainable QTCL parameters.",
    )
    parser.add_argument(
        "--qtcl-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay applied to the quantum TCL optimizer.",
    )
    parser.add_argument(
        "--qtcl-betas",
        type=float,
        nargs=2,
        metavar=("BETA1", "BETA2"),
        default=(0.9, 0.999),
        help="Beta coefficients for the quantum TCL AdamW optimizer.",
    )
    parser.add_argument(
        "--qtcl-batchnorm",
        action="store_true",
        help="Retain BatchNorm2d layers before and after each QMTL block (default: disabled).",
    )
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of training data to use.")
    parser.add_argument(
        "--val-fraction", type=float, default=1.0, help="Fraction of validation data to use as a proxy for testing."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. When --seeds is omitted, runs use seed, seed+1, ..., seed+num-seeds-1.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of consecutive seeds to run when --seeds is not provided.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of random seeds. Overrides --num-seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory where checkpoints, CSV summaries, and plots are stored.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--lr-step-size", type=int, default=40, help="Step size (in epochs) for learning rate decay.")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Multiplicative factor of learning rate decay.")
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="Distributed backend to use with torchrun (e.g., nccl, gloo).",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Number of initial epochs to skip SGD updates on the backbone while QMTL adapts.",
    )
    args = parser.parse_args()
    args.qtcl_betas = tuple(args.qtcl_betas)
    selective_scopes = (
        int(args.train_final_layer_only)
        + int(args.train_classifier_only)
        + int(args.train_last_classifier_fc_layers is not None)
    )
    if selective_scopes > 1:
        raise ValueError(
            "Specify at most one of --train-final-layer-only, --train-classifier-only, "
            "and --train-last-classifier-fc-layers."
        )
    if args.train_last_classifier_fc_layers is not None and args.train_last_classifier_fc_layers <= 0:
        raise ValueError("train-last-classifier-fc-layers must be a positive integer.")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze-backbone-epochs must be a non-negative integer.")

    resolved_seeds = resolve_run_seeds(args)
    device = init_distributed(args)
    main_process = is_main_process(args)
    args.output_dir = Path(args.output_dir)

    if main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_log_path = enable_run_log(args.output_dir)
        location = format_device_for_log(device)
        if args.distributed:
            print(
                f"Distributed training enabled "
                f"| world_size={args.world_size} | rank={args.rank} | device={location}"
            )
        else:
            print(f"Using device: {location}")
        print(f"Run log: {run_log_path}")
        print(f"Resolved seeds: {resolved_seeds}")
        write_experiment_config(args.output_dir, args, resolved_seeds)

    run_records: List[Dict[str, Any]] = []
    try:
        for run_index, run_seed in enumerate(resolved_seeds, start=1):
            record = run_single_seed(
                args=args,
                run_seed=run_seed,
                run_index=run_index,
                total_runs=len(resolved_seeds),
                device=device,
                main_process=main_process,
                resolved_seeds=resolved_seeds,
            )
            if main_process:
                run_records.append(record)
    finally:
        cleanup_distributed(args)

    if main_process and run_records:
        write_seed_summaries(args.output_dir, run_records)
        write_aggregate_summary(args.output_dir, run_records)
        write_aggregate_training_log(args.output_dir, run_records)

        best_val_accs = [float(record["best_val_acc"]) for record in run_records]
        final_val_accs = [float(record["final_val_acc"]) for record in run_records]
        best_mean, best_std = compute_mean_std(best_val_accs)
        final_mean, final_std = compute_mean_std(final_val_accs)
        print(
            f"Aggregate over {len(run_records)} seed(s) "
            f"| best_val_acc={best_mean * 100:.2f}% +/- {best_std * 100:.2f}% "
            f"| final_val_acc={final_mean * 100:.2f}% +/- {final_std * 100:.2f}%"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        raise SystemExit(130)
