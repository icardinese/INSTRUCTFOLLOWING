"""Proper PSR: direction (z_attr) is jointly gradient-trained alongside the gate, from Nokia's actual
default init -- not conceptor-projected, not warm-started. This is the paper-faithful design.

Note this file has no correction formula of its own -- it reuses steering/psr/gate.py's
forward_with_gate_hook directly, passing this trainable `direction` in. "Proper" and "conceptor"
(fixed-vector) are mechanically identical injection-wise; they only differ in where direction comes
from, which is exactly what this file's narrow job is.
"""
import torch


def init_direction(hidden_size: int, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Matches nn.Linear(hidden_size, 1, bias=False)'s actual default init (Kaiming-uniform,
    bound 1/sqrt(hidden_size)) -- Nokia's reference implementation initializes z_attr this way,
    deliberately not warm-started from any existing direction."""
    bound = 1.0 / (hidden_size ** 0.5)
    return torch.empty(hidden_size, device=device, dtype=dtype).uniform_(-bound, bound).requires_grad_(True)
