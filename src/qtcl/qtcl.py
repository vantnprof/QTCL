from typing import Tuple, Optional, Dict, Any
import warnings
import torch
import torch.nn as nn
import pennylane as qml
from src.qtcl.ansatz import ANSATZ_REGISTRY


class QTCL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int] = (2, 2),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        bias: bool = False,
        out_height: Optional[int] = None,
        out_width: Optional[int] = None,
        n_qubits: int = 8,
        n_layers: int = 2,
        F: int = None,
        shots: int = None,
        ansatz_type: str = "HEA",
        ansatz_kwargs: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        learnable_alpha: bool = False,
    ):
        super(QTCL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.out_height = out_height
        self.out_width = out_width
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.F = F if F else n_qubits
        self.shots = shots
        self.ansatz_type = ansatz_type
        self.ansatz_kwargs = ansatz_kwargs or {}
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], received {alpha}.")
        self.learnable_alpha = learnable_alpha
        if self.learnable_alpha:
            init_alpha = float(alpha)
            eps = 1e-6
            init_alpha = min(max(init_alpha, eps), 1.0 - eps)
            logit = torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
            self._alpha_logit = nn.Parameter(logit)
            self.register_buffer("alpha", None)
        else:
            self._alpha_logit = None
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

        self.k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        self.s = self.stride if isinstance(self.stride, int) else self.stride[0]
        self.p = self.padding if isinstance(self.padding, int) else self.padding[0]

        self.U1 = None
        self.U2 = None
       
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=self.shots)

        if self.ansatz_type not in ANSATZ_REGISTRY:
            raise ValueError(f"Unknown ansatz_type '{self.ansatz_type}'. Available keys: {list(ANSATZ_REGISTRY.keys())}")

        builder = ANSATZ_REGISTRY[self.ansatz_type]["builder"]
        qnode, weight_shapes = builder(
            dev=self.dev,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            F=self.F,
            **self.ansatz_kwargs,
        )

        self.B = nn.Linear(self.in_channels, self.n_qubits, bias=self.bias)
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        # self.batched_q_layer = torch.func.vmap(self.q_layer)
        self.q_feature_dim = getattr(self.q_layer, "output_dim", self.F)
        if self.q_feature_dim != self.F:
            warnings.warn(
                f"Quantum layer output dimension ({self.q_feature_dim}) differs from requested F ({self.F}). "
                "Using the reported output dimension for the classical projection.",
                RuntimeWarning,
            )
        self.A = nn.Linear(self.q_feature_dim, out_channels, bias=self.bias)

        if self.q_feature_dim == self.n_qubits:
            self.classical_proj: nn.Module = nn.Identity()
        else:
            self.classical_proj = nn.Linear(self.n_qubits, self.q_feature_dim, bias=False)

        self.is_initialized = False
        self._input_height: Optional[int] = None
        self._input_width: Optional[int] = None
        self.H_out: Optional[int] = None
        self.W_out: Optional[int] = None
        self._last_classical_mean: Optional[float] = None
        self._last_quantum_mean: Optional[float] = None

    def _init_weights(self, H: int, W: int, dtype: torch.dtype, device: torch.device, C: int = None):
        if C is not None and C != self.in_channels:
            self.in_channels = C
            self.B = nn.Linear(self.in_channels, self.n_qubits, bias=self.bias)

        self.B = self.B.to(device=device, dtype=dtype)
        self.A = self.A.to(device=device, dtype=dtype)
        self.classical_proj = self.classical_proj.to(device=device, dtype=dtype)
        H_out = self.out_height if self.out_height is not None else (H + 2 * self.p - self.k) // self.s + 1
        W_out = self.out_width if self.out_width is not None else (W + 2 * self.p - self.k) // self.s + 1
        self.H_out = H_out
        self.W_out = W_out
        self._input_height = H
        self._input_width = W

        self.U1 = nn.Parameter(torch.randn(self.H_out, H, dtype=dtype, device=device) * 0.5)
        self.U2 = nn.Parameter(torch.randn(self.W_out, W, dtype=dtype, device=device) * 0.5)

    
    def _pqc_eval(self, enc: torch.Tensor) -> torch.Tensor:
        """Avoid broadcasted tapes for shots>0 by evaluating per-sample."""
        if self.shots and self.shots > 0:
            outs = []
            for i in range(enc.shape[0]):
                outs.append(self.q_layer(enc[i]))        # (F,)
            return torch.stack(outs, dim=0)              # (N, F)
        else:
            return self.q_layer(enc)                     # (N, F)

    def _resolve_alpha(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.learnable_alpha and self._alpha_logit is not None:
            return torch.sigmoid(self._alpha_logit).to(dtype=dtype, device=device)
        if self.alpha is None:
            raise RuntimeError("Alpha buffer is not initialized.")
        return self.alpha.to(dtype=dtype, device=device)

    def forward(self, x):
        B, C, H, W = x.shape

        needs_reinit = (
            not self.is_initialized
            or self.U1 is None
            or self.U2 is None
            or self.U1.shape[1] != H
            or self.U2.shape[1] != W
            or self.in_channels != C
            or (self.out_height is not None and self.H_out != self.out_height)
            or (self.out_width is not None and self.W_out != self.out_width)
        )

        if needs_reinit:
            self._init_weights(H, W, x.dtype, x.device, C)
            self.is_initialized = True

        x = x.contiguous()

        out = torch.einsum('bchw,oh->bcow', x, self.U1)

        out = torch.einsum('bchw,ow->bcho', out, self.U2)

        z = out.permute(0, 2, 3, 1).reshape(-1, C)

        linear_out = self.B(z)  # (B*H_out*W_out, n_qubits)

        classical_features = self.classical_proj(linear_out)
        self._last_classical_mean = float(classical_features.mean().detach().cpu())

        q_inputs = torch.tanh(linear_out)

        q_features = self._pqc_eval(q_inputs)

        q_features = q_features.reshape(q_inputs.shape[0], self.q_feature_dim)
        self._last_quantum_mean = float(q_features.mean().detach().cpu())

        alpha = self._resolve_alpha(dtype=q_features.dtype, device=q_features.device)

        mixed_features = torch.lerp(classical_features, q_features, alpha)

        q_out = self.A(mixed_features)

        Z = q_out.view(B, self.H_out, self.W_out, self.out_channels)

        Z = Z.permute(0, 3, 1, 2)

        return Z

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}), n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, F={self.F}, shots={self.shots}, "
                f"ansatz_type='{self.ansatz_type}', alpha={self._alpha_repr()})")

    def _alpha_repr(self) -> str:
        if self.learnable_alpha and self._alpha_logit is not None:
            with torch.no_grad():
                return f"learnable({torch.sigmoid(self._alpha_logit).item():.4f})"
        if self.alpha is not None:
            return f"fixed({float(self.alpha):.4f})"
        return "uninitialized"

    def __str__(self):
        return str(self.__repr__())
    

if __name__ == '__main__':
    model = QTCL(
        in_channels=4,
        out_channels=4,
        kernel_size=(2, 2),
        n_qubits=8,
        n_layers=2,
        F=8,
        shots=None
    )

    # --- Create dummy inputs (not needed to print circuit but for completeness)
    dummy_input = torch.randn(1, 4, 4, 4)

    # --- Access the wrapped QNode from the TorchLayer
    qnode = model.q_layer.qnode

    print("\n=== Quantum Circuit Structure (n_qubits=8, n_layers=2) ===\n")
    print(qml.draw(qnode)(torch.zeros(8), torch.rand((2, 8))))
    print("\n==========================================================\n")
