"""Closed-form conceptor construction. C = R(R + alpha^-2 I)^-1 -- a soft projection matrix built
directly from data, no gradient descent. Every conceptor variant (fixed-vector, matrix, selfproj)
builds C the same way; they only differ in what they DO with it afterward.
"""
import torch


def compute_conceptor_from_correlation(r: torch.Tensor, alpha: float) -> torch.Tensor:
    """C = R(R + alpha^-2 I)^-1 given a precomputed correlation matrix R (d, d).

    Split out from compute_conceptor so callers that can't afford to materialize the raw (N, d)
    activation stack can accumulate R incrementally instead. This matters at all-layer scale: the
    bipolar pool is ~10.9k response-token rows, so holding it for all 28 layers would be ~8.8GB,
    whereas 28 correlation matrices are 28 * 3584^2 * 4B ~= 1.4GB and can be accumulated in one
    pass over the data. Mathematically identical -- compute_conceptor just forms R then calls this.
    """
    identity = torch.eye(r.shape[0], device=r.device, dtype=r.dtype)
    return r @ torch.linalg.inv(r + (alpha ** -2) * identity)


def compute_conceptor(activations: torch.Tensor, alpha: float) -> torch.Tensor:
    """activations: (N, d) pooled bipolar activations (steering/psr/data.py's pool_bipolar_activations).
    alpha (aperture): controls how tightly C clings to R's dominant directions -- small alpha = more
    selective attenuation, large alpha -> C approaches the identity matrix."""
    n, _d = activations.shape
    r = (activations.T @ activations) / n
    return compute_conceptor_from_correlation(r, alpha)


def project_direction(conceptor: torch.Tensor, base_direction: torch.Tensor) -> torch.Tensor:
    """Reshapes an existing fixed direction (e.g. a diff-in-means vector) through the conceptor's
    subspace -- this is what makes the "fixed-vector" conceptor variant's direction closed-form
    rather than gradient-trained. Returns a unit vector."""
    v = conceptor @ base_direction
    return v / v.norm()
