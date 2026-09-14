"""Direction-projection steering: pins a hidden state's projection along a direction to an EXACT
target value, rather than adding a fixed delta (additive steering) or interpolating toward a matrix
projection (task_matrix). Ported from a TransformerLens-hook-signature original; generalized from
the original's batch=1 assumption (it used .squeeze(0)/.unsqueeze(0)) to arbitrary batch/seq via
reshape -- this codebase never actually runs batch>1, but the reshape costs nothing and removes a
silent assumption.
"""
from typing import Callable

import torch


def adjust_vectors(v: torch.Tensor, u: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
    """Adjusts rows of v so their projection along the unit vector u equals target_values.

    v: (n, d). u: (d,) unit vector. target_values: (n,) desired projections. Returns (n, d)."""
    current_projections = v @ u
    delta = target_values - current_projections
    return v + delta[:, None] * u


def make_projection_hook(direction: torch.Tensor, value_along_direction: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """value_along_direction must broadcast to (batch*seq,) -- either a single scalar target for
    every position, or one target per position matching hidden's flattened batch*seq count."""
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, d = hidden.shape
        u = direction.to(hidden.dtype).to(hidden.device)
        u = u / u.norm()  # adjust_vectors assumes a UNIT vector; normalize here so callers don't have to remember to
        v = hidden.reshape(-1, d)
        targets = value_along_direction.to(hidden.dtype).to(hidden.device)
        if targets.numel() == 1:
            targets = targets.expand(v.shape[0])
        adjusted = adjust_vectors(v, u, targets)
        return adjusted.reshape(batch, seq, d)

    return hook_fn
