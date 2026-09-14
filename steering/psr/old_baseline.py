"""The pre-fidelity-fix S-PSR/A-PSR baseline: a plain ReLU probe producing a token-specific
coefficient on a fixed direction. Kept as its own file distinct from steering/psr/proper/ and
steering/psr/conceptor/ -- this is the ORIGINAL, simpler design (offline MSE, injection-layer-only,
no regularization), retained as a baseline comparison point, not the fidelity-fixed training regime.
"""
from typing import Callable

import torch


class PSRProbe(torch.nn.Module):
    """Single-layer ReLU probe producing a token-specific steering coefficient (S-PSR)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(hidden))


class MultiLayerPSRProbe(torch.nn.Module):
    """One PSRProbe per layer, trained jointly (A-PSR). ModuleDict keys are strings since PyTorch
    module dicts don't accept int keys directly -- layer_indices is kept separately for convenience."""

    def __init__(self, hidden_size: int, layer_indices: list[int]):
        super().__init__()
        self.layer_indices = layer_indices
        self.probes = torch.nn.ModuleDict({str(l): PSRProbe(hidden_size) for l in layer_indices})

    def forward(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.probes[str(layer_idx)](hidden)


def make_psr_hook(direction: torch.Tensor, probe: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        lam = probe(hidden.float()).to(hidden.dtype)  # (batch, seq, 1), token-specific coefficient
        return hidden + lam * direction.to(hidden.dtype).to(hidden.device)

    return hook_fn


def make_multi_psr_hooks(
    directions: dict[int, torch.Tensor], probe: MultiLayerPSRProbe
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Builds the per-layer hook dict for A-PSR, for use with steering.hooks.multi_steering_hook."""
    hooks = {}
    for layer_idx, direction in directions.items():

        def _make_hook(l, d):
            def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
                lam = probe(l, hidden.float()).to(hidden.dtype)
                return hidden + lam * d.to(hidden.dtype).to(hidden.device)

            return hook_fn

        hooks[layer_idx] = _make_hook(layer_idx, direction)
    return hooks
