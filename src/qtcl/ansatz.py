from typing import Callable, Dict, Tuple, Optional, Any
import inspect

import pennylane as qml


QNodeFactory = Callable[..., Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]]


def _ensure_qaoa_shape_accepts_wires() -> None:
    """Patch :func:`qml.QAOAEmbedding.shape` to accept ``wires`` for older PennyLane versions."""
    try:
        shape_fn = qml.QAOAEmbedding.shape
    except AttributeError:
        return

    signature = inspect.signature(shape_fn)
    if "wires" in signature.parameters:
        # Current PennyLane already supports the wires kwarg.
        return

    def _shape_wrapper(*args, **kwargs):
        if "wires" in kwargs and "n_wires" not in kwargs:
            wires = kwargs.pop("wires")
            if isinstance(wires, int):
                kwargs["n_wires"] = wires
            else:
                try:
                    kwargs["n_wires"] = len(wires)
                except TypeError as exc:
                    raise TypeError(
                        "Cannot infer n_wires from the provided wires argument."
                    ) from exc
        return shape_fn(*args, **kwargs)

    qml.QAOAEmbedding.shape = staticmethod(_shape_wrapper)


_ensure_qaoa_shape_accepts_wires()

_QAOA_SUPPORTS_INITIAL_LAYER = "initial_layer" in inspect.signature(qml.QAOAEmbedding).parameters


def _qaoa_weight_shape(n_layers: int, n_qubits: int):
    """Return the expected QAOA weight shape across PennyLane versions."""
    try:
        return qml.QAOAEmbedding.shape(n_layers=n_layers, n_wires=n_qubits)
    except TypeError:
        return qml.QAOAEmbedding.shape(n_layers=n_layers, wires=list(range(n_qubits)))


def _validate_measurements(F: int, n_qubits: int) -> None:
    if F > n_qubits:
        raise ValueError(f"Requested F={F} expectation values but only {n_qubits} qubits are available.")


def _measure_expectations(F: int):
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(F))


def _two_qubit_tensor_block(weights, wires):
    """Lightweight two-qubit tensor block used by MPS/MERA variants."""
    if len(wires) != 2:
        raise ValueError("Two-qubit tensor block expects exactly two wires.")
    qml.IsingXX(weights[0], wires=wires)
    qml.IsingYY(weights[1], wires=wires)
    qml.IsingZZ(weights[2], wires=wires)


def build_hardware_efficient_ansatz(
    dev,
    n_qubits: int,
    n_layers: int,
    F: int,
    *,
    rotation: str = "X",
) -> Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]:
    """Hardware-efficient ansatz: Angle embedding + hardware-efficient entanglers."""
    _validate_measurements(F, n_qubits)
    wires = list(range(n_qubits))
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)
        qml.BasicEntanglerLayers(weights, wires=wires)
        return _measure_expectations(F)

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes


def build_data_reuploading_ansatz(
    dev,
    n_qubits: int,
    n_layers: int,
    F: int,
    *,
    rotation: str = "Y",
) -> Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]:
    """Data re-uploading ansatz: re-encode inputs before every trainable layer using strongly entangling blocks."""
    _validate_measurements(F, n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)
            qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=wires)
        return _measure_expectations(F)

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return circuit, weight_shapes


def build_qaoa_embedding_ansatz(
    dev,
    n_qubits: int,
    n_layers: int,
    F: int,
    *,
    local_field: str = "Y",
    initial_layer: Optional[bool] = None,
) -> Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]:
    """QAOA-inspired embedding ansatz following :func:`qml.QAOAEmbedding`.

    The template applies alternating problem and mixer unitaries with weights of shape
    ``(n_layers, 2)`` and expects feature vectors whose trailing dimension matches
    the number of wires, as described in the PennyLane tutorial.
    """
    _validate_measurements(F, n_qubits)
    wires = list(range(n_qubits))

    weight_shape = _qaoa_weight_shape(n_layers=n_layers, n_qubits=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qaoa_kwargs = {
            "features": inputs,
            "weights": weights,
            "wires": wires,
            "local_field": local_field,
        }
        if initial_layer is not None:
            if _QAOA_SUPPORTS_INITIAL_LAYER:
                qaoa_kwargs["initial_layer"] = initial_layer
            elif initial_layer:
                raise ValueError(
                    "initial_layer=True requires a PennyLane version that supports the initial_layer argument."
                )
        qml.QAOAEmbedding(**qaoa_kwargs)
        return _measure_expectations(F)

    weight_shapes = {"weights": weight_shape}
    return circuit, weight_shapes


def build_mps_ansatz(
    dev,
    n_qubits: int,
    n_layers: int,
    F: int,
    *,
    n_block_wires: int = 2,
    n_params_block: int = 3,
    rotation: str = "Y",
) -> Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]:
    """Matrix Product State ansatz using :func:`qml.MPS`."""
    if n_qubits < n_block_wires:
        raise ValueError(f"MPS ansatz requires at least {n_block_wires} qubits.")
    if n_block_wires != 2:
        raise ValueError("Current MPS block supports only n_block_wires=2.")
    _validate_measurements(F, n_qubits)
    wires = list(range(n_qubits))
    n_blocks = n_qubits - n_block_wires + 1

    template_shape = (n_layers, n_blocks, n_params_block)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)
        for layer in range(n_layers):
            qml.MPS(
                wires=wires,
                n_block_wires=n_block_wires,
                block=_two_qubit_tensor_block,
                n_params_block=n_params_block,
                template_weights=weights[layer],
            )
        return _measure_expectations(F)

    weight_shapes = {"weights": template_shape}
    return circuit, weight_shapes


def build_mera_ansatz(
    dev,
    n_qubits: int,
    n_layers: int,
    F: int,
    *,
    n_block_wires: int = 2,
    n_params_block: int = 3,
    rotation: str = "Y",
) -> Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]:
    """Multiscale MERA ansatz using :func:`qml.MERA`."""
    if n_qubits < n_block_wires:
        raise ValueError(f"MERA ansatz requires at least {n_block_wires} qubits.")
    if n_block_wires != 2:
        raise ValueError("Current MERA block supports only n_block_wires=2.")
    _validate_measurements(F, n_qubits)
    wires = list(range(n_qubits))
    n_tensors = (n_qubits - 1) + (n_qubits - n_block_wires)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)
        for layer in range(n_layers):
            qml.MERA(
                wires=wires,
                n_block_wires=n_block_wires,
                block=_two_qubit_tensor_block,
                n_params_block=n_params_block,
                template_weights=weights[layer],
            )
        return _measure_expectations(F)

    weight_shapes = {"weights": (n_layers, n_tensors, n_params_block)}
    return circuit, weight_shapes


ANSATZ_REGISTRY: Dict[str, Dict[str, Any]] = {
    "HEA": {
        "builder": build_hardware_efficient_ansatz,
        "desc": "Hardware-efficient layers with ring entanglers.",
    },
    "SLE": {
        "builder": build_data_reuploading_ansatz,
        "desc": "Strongly entangling data re-uploading ansatz.",
    },
    "QAOA": {
        "builder": build_qaoa_embedding_ansatz,
        "desc": "QAOA-inspired feature embedding.",
    },
    "MPS": {
        "builder": build_mps_ansatz,
        "desc": "Nearest-neighbour MPS tensor network (bond dimension 2).",
    },
    "MERA": {
        "builder": build_mera_ansatz,
        "desc": "Multiscale MERA-style tensor network.",
    },
}


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    n_qubits = 8
    n_layers = 1
    F = 8
    dev = qml.device("default.qubit", wires=n_qubits)

    ansatz_order = ["HEA", "SLE", "QAOA", "MPS", "MERA"]
    dtype = torch.float64
    test_input = torch.randn(n_qubits, dtype=dtype)

    print("=== Sanity check for registered ansätze ===")
    for name in ansatz_order:
        if name not in ANSATZ_REGISTRY:
            print(f"[{name}] missing from registry, skipping.")
            continue

        builder = ANSATZ_REGISTRY[name]["builder"]
        try:
            qnode, weight_shapes = builder(
                dev=dev,
                n_qubits=n_qubits,
                n_layers=n_layers,
                F=F,
            )

            weight_shape = weight_shapes["weights"]
            weights = torch.randn(*weight_shape, dtype=dtype)
            outputs = torch.as_tensor(qnode(test_input, weights))

            print(f"[{name}] weight shape={weight_shape}, output shape={outputs.shape}")
        except Exception as err:
            print(f"[{name}] failed with: {err}")