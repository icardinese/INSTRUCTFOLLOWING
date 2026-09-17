"""Companion to steering/const/direction.py's compute_diff_mean_direction -- same base/instr
activations, but this project's Stolfo-projection method needs one more number Const doesn't:
the target projection value the correction aims for."""
import torch


def compute_target_projection(instr_activations: torch.Tensor, direction: torch.Tensor) -> float:
    """instr_activations: (N, d) -- the INSTRUCTED condition's own activations (not base, not the
    diff). direction: unit vector (d,), from compute_diff_mean_direction. Returns avg_proj exactly
    as format/find_best_layer.py computes it: proj = hs_instr @ instr_dir; avg_proj = proj.mean()."""
    proj = instr_activations.float() @ direction.float()
    return proj.mean().item()
