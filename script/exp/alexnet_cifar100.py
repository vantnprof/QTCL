import argparse
import copy
import csv
import json
import math
import os
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

    classifier = model.classifier
    linear_indices = [idx for idx, layer in enumerate(classifier) if isinstance(layer, nn.Linear)]
    if not linear_indices:
        raise ValueError("Model classifier does not contain a linear layer to replace.")

    first_linear_idx = linear_indices[0]
    first_linear = classifier[first_linear_idx]

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
    if debug:
        print(
            "[replace_1st_fc_with_tcl] Detected first linear layer "
            f"(index={first_linear_idx}) in_features={in_features}, out_features={first_linear.out_features}"
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

    prefix_layers = list(classifier[:first_linear_idx])
    rest_layers = list(classifier[first_linear_idx + 1 :])

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
        *prefix_layers,
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

    classifier = model.classifier
    linear_indices = [idx for idx, layer in enumerate(classifier) if isinstance(layer, nn.Linear)]
    if len(linear_indices) < 2:
        raise ValueError("Model classifier must expose at least two linear layers.")

    first_linear_idx, second_linear_idx = linear_indices[:2]
    first_linear = classifier[first_linear_idx]

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
        for idx, layer in enumerate(classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")
        print(
            "[replace_1st_2nd_fc_with_tcl] Resolved first linear in_features="
            f"{in_features} (channels={in_channels}, height={in_height}, width={in_width})"
        )

    prefix_layers = list(classifier[:first_linear_idx])
    between_layers = list(classifier[first_linear_idx + 1 : second_linear_idx])
    tail_layers = list(classifier[second_linear_idx + 1 :])

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

    bn2_out = nn.BatchNorm2d(second_out_channels).to(device=device, dtype=dtype)
    tcl2 = TCL(
        in_channels=first_out_channels,
        out_channels=second_out_channels,
        out_height=second_out_height,
        out_width=second_out_width,
    ).to(device=device, dtype=dtype)

    new_classifier_layers: List[nn.Module] = [*prefix_layers, nn.Unflatten(1, (in_channels, in_height, in_width))]
    if bn1_in is not None:
        new_classifier_layers.append(bn1_in)
    new_classifier_layers.append(tcl1)
    if bn1_out is not None:
        new_classifier_layers.append(bn1_out)
    new_classifier_layers.extend(between_layers)
    new_classifier_layers.append(tcl2)
    if bn2_out is not None:
        new_classifier_layers.append(bn2_out)
    new_classifier_layers.append(nn.Flatten())
    new_classifier_layers.extend(tail_layers)

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

    classifier = model.classifier
    linear_indices = [idx for idx, layer in enumerate(classifier) if isinstance(layer, nn.Linear)]
    if not linear_indices:
        raise ValueError("Model classifier does not contain a linear layer to replace.")

    first_linear_idx = linear_indices[0]
    first_layer = classifier[first_linear_idx]

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

    prefix_layers = list(classifier[:first_linear_idx])
    rest_layers = list(classifier[first_linear_idx + 1 :])

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if debug:
        print("[replace_1st_fc_with_qtcl] Original classifier:")
        for idx, layer in enumerate(classifier):
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
        *prefix_layers,
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

    classifier = model.classifier
    linear_indices = [idx for idx, layer in enumerate(classifier) if isinstance(layer, nn.Linear)]
    if len(linear_indices) < 2:
        raise ValueError("Model classifier must expose at least two linear layers.")

    first_linear_idx, second_linear_idx = linear_indices[:2]
    first_linear = classifier[first_linear_idx]

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
        for idx, layer in enumerate(classifier):
            print(f"  [{idx}]: {layer.__class__.__name__} -> {layer}")

    prefix_layers = list(classifier[:first_linear_idx])
    between_layers = list(classifier[first_linear_idx + 1 : second_linear_idx])
    tail_layers = list(classifier[second_linear_idx + 1 :])

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

    new_classifier_layers: List[nn.Module] = [*prefix_layers, nn.Unflatten(1, (in_channels, in_height, in_width))]
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
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    model.avgpool = nn.AdaptiveAvgPool2d((3, 3))
    model.classifier[1] = nn.Linear(256 * 3 * 3, 4096, bias=True)
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


def _subset_dataset(dataset, fraction: float, seed: Optional[int]):
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError("fraction must be in the interval (0, 1].")

    subset_size = max(1, math.ceil(fraction * len(dataset)))
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


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
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    quantum_optimizer: Optional[optim.Optimizer],
    freeze_backbone: bool,
    device: torch.device,
    epoch_desc: str,
    show_progress: bool,
    distributed: bool,
) -> Tuple[float, float, bool]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    backbone_stepped = False
    iterator = tqdm(dataloader, desc=epoch_desc, leave=False, disable=not show_progress)
    for images, targets in iterator:
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

        if show_progress and total > 0:
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
    iterator = tqdm(dataloader, desc=epoch_desc, leave=False, disable=not show_progress)
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        running_correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if show_progress and total > 0:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-100 using PyTorch.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate for SGD optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for SGD optimizer.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    parser.add_argument(
        "--model",
        type=str,
        default="alexnet",
        choices=[
            "alexnet",
            "alexnet_tcl",
            "alexnet_tcl1",
            "alexnet_tcl12",
            "alexnet_qtcl",
            "alexnet_qtcl1",
            "alexnet_qtcl12",
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
            "--model alexnet_tcl1/alexnet_tcl12/alexnet_qtcl1/alexnet_qtcl12."
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
            "--model alexnet_tcl1/alexnet_tcl12/alexnet_qtcl1/alexnet_qtcl12."
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
            "--model alexnet_tcl1/alexnet_tcl12/alexnet_qtcl1/alexnet_qtcl12."
        ),
    )
    parser.add_argument(
        "--tcl2-out-channels",
        type=int,
        default=32,
        help="Output channel count for second replacement when using --model alexnet_tcl12/alexnet_qtcl12.",
    )
    parser.add_argument(
        "--tcl2-out-height",
        type=int,
        default=8,
        help="Output height for second replacement when using --model alexnet_tcl12/alexnet_qtcl12.",
    )
    parser.add_argument(
        "--tcl2-out-width",
        type=int,
        default=8,
        help="Output width for second replacement when using --model alexnet_tcl12/alexnet_qtcl12.",
    )
    parser.add_argument(
        "--tcl-debug",
        action="store_true",
        help="Enable verbose debugging for TCL/QTCL replacements.",
    )
    parser.add_argument(
        "--qtcl-n-qubits",
        type=int,
        default=8,
        help="Number of qubits for quantum TCL replacements.",
    )
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
        help="Override qubit count for the second quantum TCL stage when using --model alexnet_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-n-layers",
        type=int,
        default=None,
        help="Override layer count for the second quantum TCL stage when using --model alexnet_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-F",
        type=int,
        default=None,
        help="Override latent feature width F for the second quantum TCL stage when using --model alexnet_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-alpha",
        type=float,
        default=None,
        help="Override the mixing coefficient alpha for the second QTCL block when using --model alexnet_qtcl12.",
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
        help="Override shots for the second quantum TCL stage when using --model alexnet_qtcl12 (0 means analytic).",
    )
    parser.add_argument(
        "--qtcl2-ansatz",
        type=str,
        default=None,
        help="Override ansatz key for the second quantum TCL stage when using --model alexnet_qtcl12.",
    )
    parser.add_argument(
        "--qtcl2-ansatz-kwargs",
        type=str,
        default=None,
        help="Override ansatz kwargs (JSON dict) for the second quantum TCL stage when using --model alexnet_qtcl12.",
    )
    parser.add_argument(
        "--qtcl-lr",
        type=float,
        default=5e-3,
        help="Learning rate for quantum TCL parameters when a dedicated optimizer is used.",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset subsampling.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory where checkpoints and training plots are stored.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--lr-step-size", type=int, default=30, help="Step size (in epochs) for learning rate decay.")
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
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze-backbone-epochs must be a non-negative integer.")

    device = init_distributed(args)
    main_process = is_main_process(args)

    args.output_dir = Path(args.output_dir)

    if main_process:
        location = f"{device.type}:{device.index}" if device.type == "cuda" else device.type
        if args.distributed:
            print(
                f"Distributed training enabled "
                f"| world_size={args.world_size} | rank={args.rank} | device={location}"
            )
        else:
            print(f"Using device: {location}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(num_classes=100, pretrained=args.pretrained, device=device)

    model_variant = args.model
    if model_variant == "alexnet_tcl":
        model_variant = "alexnet_tcl1"
    if model_variant == "alexnet_qtcl":
        model_variant = "alexnet_qtcl1"

    first_stage_variants = {"alexnet_tcl1", "alexnet_tcl12", "alexnet_qtcl1", "alexnet_qtcl12"}
    second_stage_variants = {"alexnet_tcl12", "alexnet_qtcl12"}

    if model_variant in first_stage_variants:
        if (
            args.tcl1_out_channels <= 0
            or args.tcl1_out_height <= 0
            or args.tcl1_out_width <= 0
        ):
            raise ValueError("First replacement output dimensions must be positive integers.")

    if model_variant in second_stage_variants:
        if (
            args.tcl2_out_channels <= 0
            or args.tcl2_out_height <= 0
            or args.tcl2_out_width <= 0
        ):
            raise ValueError("Second replacement output dimensions must be positive integers.")

    qtcl_first_kwargs: Optional[Dict[str, Any]] = None
    qtcl_second_kwargs: Optional[Dict[str, Any]] = None
    if model_variant in {"alexnet_qtcl1", "alexnet_qtcl12"}:
        if args.qtcl_n_qubits <= 0:
            raise ValueError("qtcl-n-qubits must be a positive integer.")
        if args.qtcl_n_layers <= 0:
            raise ValueError("qtcl-n-layers must be a positive integer.")
        if args.qtcl_shots < 0:
            raise ValueError("qtcl-shots must be non-negative (0 means analytic expectation).")
        if args.qtcl_F is not None and args.qtcl_F <= 0:
            raise ValueError("qtcl-F must be a positive integer when provided.")
        if not 0.0 <= args.qtcl_alpha <= 1.0:
            raise ValueError("qtcl-alpha must lie in the interval [0, 1].")

        try:
            qtcl_ansatz_kwargs_loaded = json.loads(args.qtcl_ansatz_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse --qtcl-ansatz-kwargs as JSON.") from exc
        if not isinstance(qtcl_ansatz_kwargs_loaded, dict):
            raise ValueError("--qtcl-ansatz-kwargs must decode to a JSON object.")

        qtcl_first_kwargs = {
            "n_qubits": args.qtcl_n_qubits,
            "n_layers": args.qtcl_n_layers,
            "F": args.qtcl_F,
            "shots": None if args.qtcl_shots == 0 else args.qtcl_shots,
            "ansatz_type": args.qtcl_ansatz,
            "ansatz_kwargs": dict(qtcl_ansatz_kwargs_loaded),
            "alpha": args.qtcl_alpha,
            "learnable_alpha": args.qtcl_learnable_alpha,
        }

        qtcl_second_kwargs = copy.deepcopy(qtcl_first_kwargs)

        if args.qtcl2_n_qubits is not None:
            if args.qtcl2_n_qubits <= 0:
                raise ValueError("qtcl2-n-qubits must be a positive integer when provided.")
            qtcl_second_kwargs["n_qubits"] = args.qtcl2_n_qubits
        if args.qtcl2_n_layers is not None:
            if args.qtcl2_n_layers <= 0:
                raise ValueError("qtcl2-n-layers must be a positive integer when provided.")
            qtcl_second_kwargs["n_layers"] = args.qtcl2_n_layers
        if args.qtcl2_F is not None:
            if args.qtcl2_F <= 0:
                raise ValueError("qtcl2-F must be a positive integer when provided.")
            qtcl_second_kwargs["F"] = args.qtcl2_F
        if args.qtcl2_shots is not None:
            if args.qtcl2_shots < 0:
                raise ValueError("qtcl2-shots must be non-negative (0 means analytic expectation).")
            qtcl_second_kwargs["shots"] = None if args.qtcl2_shots == 0 else args.qtcl2_shots
        if args.qtcl2_ansatz is not None:
            qtcl_second_kwargs["ansatz_type"] = args.qtcl2_ansatz
        if args.qtcl2_ansatz_kwargs is not None:
            try:
                qtcl2_ansatz_kwargs_loaded = json.loads(args.qtcl2_ansatz_kwargs)
            except json.JSONDecodeError as exc:
                raise ValueError("Failed to parse --qtcl2-ansatz-kwargs as JSON.") from exc
            if not isinstance(qtcl2_ansatz_kwargs_loaded, dict):
                raise ValueError("--qtcl2-ansatz-kwargs must decode to a JSON object.")
            qtcl_second_kwargs["ansatz_kwargs"] = dict(qtcl2_ansatz_kwargs_loaded)
        if args.qtcl2_alpha is not None:
            if not 0.0 <= args.qtcl2_alpha <= 1.0:
                raise ValueError("qtcl2-alpha must lie in the interval [0, 1].")
            qtcl_second_kwargs["alpha"] = args.qtcl2_alpha
        if args.qtcl2_learnable_alpha is not None:
            qtcl_second_kwargs["learnable_alpha"] = args.qtcl2_learnable_alpha

    if model_variant == "alexnet_tcl1":
        model = replace_1st_fc_with_tcl(
            model,
            out_channels=args.tcl1_out_channels,
            out_height=args.tcl1_out_height,
            out_width=args.tcl1_out_width,
            debug=args.tcl_debug and main_process,
        )
    elif model_variant == "alexnet_tcl12":
        model = replace_1st_2nd_fc_with_tcl(
            model,
            first_out_channels=args.tcl1_out_channels,
            first_out_height=args.tcl1_out_height,
            first_out_width=args.tcl1_out_width,
            second_out_channels=args.tcl2_out_channels,
            second_out_height=args.tcl2_out_height,
            second_out_width=args.tcl2_out_width,
            debug=args.tcl_debug and main_process,
        )
    elif model_variant == "alexnet_qtcl1":
        model = replace_1st_fc_with_qtcl(
            model,
            out_channels=args.tcl1_out_channels,
            out_height=args.tcl1_out_height,
            out_width=args.tcl1_out_width,
            use_batchnorm=True,
            qmtl_kwargs=qtcl_first_kwargs,
            debug=args.tcl_debug and main_process,
        )
    elif model_variant == "alexnet_qtcl12":
        model = replace_1st_2nd_fc_with_qtcl(
            model,
            first_out_channels=args.tcl1_out_channels,
            first_out_height=args.tcl1_out_height,
            first_out_width=args.tcl1_out_width,
            second_out_channels=args.tcl2_out_channels,
            second_out_height=args.tcl2_out_height,
            second_out_width=args.tcl2_out_width,
            use_batchnorm=args.qtcl_batchnorm,
            first_qmtl_kwargs=qtcl_first_kwargs,
            second_qmtl_kwargs=qtcl_second_kwargs,
            debug=args.tcl_debug and main_process,
        )
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    dummy_forward_pass(model, device, log=main_process)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.train_fraction,
        args.val_fraction,
        args.seed,
        args.distributed,
        args.rank,
        args.world_size,
    )
    if main_process:
        print(f"Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")

    param_model = model.module if isinstance(model, DDP) else model
    backbone_params, quantum_params = split_qtcl_parameter_groups(param_model)
    if not backbone_params:
        raise ValueError("Backbone parameter group is empty; expected at least one non-quantum parameter.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        backbone_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False,
    )
    quantum_optimizer: Optional[optim.Optimizer] = None
    if quantum_params:
        quantum_optimizer = optim.AdamW(
            quantum_params,
            lr=args.qtcl_lr,
            betas=args.qtcl_betas,
            weight_decay=args.qtcl_weight_decay,
        )
    if args.lr_step_size <= 0:
        raise ValueError("lr_step_size must be a positive integer.")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

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
    figure_path: Optional[Path] = None

    try:
        for epoch in range(1, args.epochs + 1):
            if args.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if args.distributed and val_sampler is not None:
                val_sampler.set_epoch(epoch)

            show_progress = (not args.no_progress) and main_process
            freeze_backbone = epoch <= args.freeze_backbone_epochs

            train_loss, train_acc, backbone_stepped = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                quantum_optimizer,
                freeze_backbone,
                device,
                epoch_desc=f"Epoch {epoch:02d} train",
                show_progress=show_progress,
                distributed=args.distributed,
            )
            log_qtcl_parameter_debug(model, epoch, main_process)
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                epoch_desc=f"Epoch {epoch:02d} val",
                show_progress=show_progress,
                distributed=args.distributed,
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
                best_checkpoint_path = args.output_dir / "checkpoint_best_acc.pth"
                checkpoint_args = {
                    k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                }
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
                    f"Epoch {epoch:02d}/{args.epochs} "
                    f"| train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}% "
                    f"| val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}% "
                    f"| best_acc={best_val_acc * 100:.2f}%{best_marker} "
                    f"| lr={optimizer.param_groups[0]['lr']:.4e}"
                )

            if backbone_stepped:
                scheduler.step()
            elif epoch <= args.freeze_backbone_epochs and main_process:
                warnings.warn(
                    "Skipped learning rate scheduler step because the backbone optimizer did not update; "
                    "this is expected while --freeze-backbone-epochs is in effect.",
                    RuntimeWarning,
                )
    finally:
        cleanup_distributed(args)

    if main_process and train_losses:
        log_path = args.output_dir / "training_log.csv"
        with log_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            for epoch_idx, (tr_loss, tr_acc, va_loss, va_acc) in enumerate(
                zip(train_losses, train_accuracies, val_losses, val_accuracies),
                start=1,
            ):
                writer.writerow([epoch_idx, tr_loss, tr_acc, va_loss, va_acc])

        final_train_loss = train_losses[-1]
        final_train_acc = train_accuracies[-1]
        final_val_loss = val_losses[-1]
        final_val_acc = val_accuracies[-1]
        summary_path = args.output_dir / "training_summary.csv"
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
                    best_val_acc,
                    best_val_acc_epoch,
                    best_val_loss,
                    best_val_loss_epoch,
                    final_train_loss,
                    final_train_acc,
                    final_val_loss,
                    final_val_acc,
                ]
            )

        figure_path = args.output_dir / "training_curves.png"
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

    if main_process:
        if best_val_acc_epoch > 0:
            print(
                f"Best validation accuracy: {best_val_acc * 100:.2f}% (epoch {best_val_acc_epoch:02d})"
            )
        else:
            print("Best validation accuracy: N/A")

        if best_val_loss_epoch > 0:
            print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_val_loss_epoch:02d})")
        else:
            print("Best validation loss: N/A")

        if best_checkpoint_path is not None:
            print(f"Best checkpoint saved to: {best_checkpoint_path}")
        else:
            print("Best checkpoint was not saved (no improvement).")

        if figure_path is not None:
            print(f"Training curves saved to: {figure_path}")
        else:
            print("Training curves were not generated.")


if __name__ == "__main__":
    main()