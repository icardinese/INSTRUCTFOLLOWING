"""Closed-form conceptor construction. C = R(R + alpha^-2 I)^-1 -- a soft projection matrix built
directly from data, no gradient descent. Every conceptor variant (fixed-vector, matrix, selfproj)
builds C the same way; they only differ in what they DO with it afterward.
"""
import torch


def compute_conceptor(activations: torch.Tensor, alpha: float) -> torch.Tensor:
    """activations: (N, d) pooled bipolar activations (steering/psr/data.py's pool_bipolar_activations).
    alpha (aperture): controls how tightly C clings to R's dominant directions -- small alpha = more
    selective attenuation, large alpha -> C approaches the identity matrix."""
    n, d = activations.shape
    r = (activations.T @ activations) / n
    identity = torch.eye(d, device=activations.device, dtype=activations.dtype)
    return r @ torch.linalg.inv(r + (alpha ** -2) * identity)


def project_direction(conceptor: torch.Tensor, base_direction: torch.Tensor) -> torch.Tensor:
    """Reshapes an existing fixed direction (e.g. a diff-in-means vector) through the conceptor's
    subspace -- this is what makes the "fixed-vector" conceptor variant's direction closed-form
    rather than gradient-trained. Returns a unit vector."""
    v = conceptor @ base_direction
    return v / v.norm()
