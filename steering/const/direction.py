"""Diff-in-means direction for constant additive steering: the mean residual-stream activation over
instructed prompts minus the mean over uninstructed ones, at a given layer. Closed-form, no training.
"""
import torch


def compute_diff_mean_direction(base_activations: torch.Tensor, instr_activations: torch.Tensor) -> torch.Tensor:
    """base_activations, instr_activations: (N, d) matched activation samples. Returns a unit vector.

    Raises rather than silently returning NaN if base and instructed activations are identical
    (direction norm exactly 0) -- dividing by a zero norm would otherwise poison every downstream
    step (steering with a NaN direction corrupts the whole generation, not just this layer) with no
    clear signal of where it went wrong."""
    direction = instr_activations.mean(0) - base_activations.mean(0)
    norm = direction.norm()
    if norm < 1e-8:
        raise ValueError(
            "diff-mean direction has ~zero norm -- base and instructed activations are "
            "indistinguishable at this layer. Check that base_prompt and terse_prompt are actually "
            "different inputs reaching the model (a common cause: a tokenizer/adapter bug that "
            "truncates or otherwise collapses the two prompts to the same tokens)."
        )
    return direction / norm
