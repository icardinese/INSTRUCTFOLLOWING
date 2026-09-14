"""Stolfo-style constant additive steering: h' = h + coeff * direction, same coefficient at every
position. No training, just calibration -- see steering/const/train.py (the old steering_const.py)
for how coeff/layer get chosen.
"""
from typing import Callable

import torch


def make_const_hook(direction: torch.Tensor, coeff: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        return hidden + coeff * direction.to(hidden.dtype).to(hidden.device)

    return hook_fn


def make_multi_const_hooks(
    directions: dict[int, torch.Tensor], coeff: float
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """A-Const: the same fixed-coefficient additive steering as S-Const, applied at every layer at once."""
    hooks = {}
    for layer_idx, direction in directions.items():

        def _make_hook(d):
            def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
                return hidden + coeff * d.to(hidden.dtype).to(hidden.device)

            return hook_fn

        hooks[layer_idx] = _make_hook(direction)
    return hooks
