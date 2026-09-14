"""Multiplicative/linear task-matrix steering: learns a matrix W such that W @ x approximates the
instructed activation for an uninstructed x, then steers via interpolation h' = h + alpha*(Wh - h)
-- genuinely different from additive (const/PSR) steering, which only ever adds a vector. Ported
from a TransformerLens-hook-signature original; math unchanged, only the hook calling convention
changed (this codebase's hooks take/return `hidden` only, no `hook` object, matching
steering/hooks.py's contract).
"""
from typing import Callable

import numpy as np
import torch


def compute_task_matrix(x_base: np.ndarray, x_instr: np.ndarray) -> np.ndarray:
    """x_base, x_instr: (N, d) uninstructed/instructed activation pairs. Returns W (d, d) solving
    x_base @ W.T ~= x_instr via least squares. Kept as numpy (not torch.linalg.lstsq) deliberately --
    a different solver could give subtly different numerical results, and this is a direct port of
    an already-used computation, not a place to introduce unreviewed numerical drift."""
    w_t, _, _, _ = np.linalg.lstsq(x_base, x_instr, rcond=None)
    return w_t.T


def make_task_matrix_hook(task_matrix: torch.Tensor, alpha: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        w = task_matrix.to(hidden.dtype).to(hidden.device)
        steered = hidden @ w.T
        return hidden + alpha * (steered - hidden)

    return hook_fn
