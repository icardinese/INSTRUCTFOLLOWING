"""Participation ratio (PR) of a conceptor matrix -- an effective-dimensionality diagnostic:

    PR = (sum_i lambda_i)^2 / sum_i (lambda_i)^2

where lambda_i are C's eigenvalues. PR ranges from 1 (C behaves like a single direction -- a rank-1
correction in disguise) up to d (C spreads weight evenly across all d dimensions, the "genuinely
multidimensional" end). This is the file steering/psr/conceptor/selfproj/logic.py's
compute_delta_scale docstring already pointed at ("see rank_diagnostic.py before trusting a given
alpha here") but that never got built during the refactor -- this fills that gap.

Only meaningful for MATRIX-BASED methods (conceptor/matrix, conceptor/selfproj): both apply C fresh
to every hidden state at inference (C @ (mu_instr - h), C @ h - h respectively), so C's effective
rank genuinely constrains the correction's expressiveness. The FIXED-VECTOR conceptor variant
(steering/psr/conceptor/direction.py's project_direction) also builds a C, but only ever uses it
once, offline, to produce a single fixed direction -- the injected correction at inference is
still exactly rank-1 regardless of C's PR (see project finding #3: "the conceptor's
multidimensionality never enters the actual injected correction"). Attaching PR to that variant's
outputs would measure a property of an intermediate computation that the final correction doesn't
actually inherit, which is why src/generate.py deliberately does NOT attach it there -- see that
file's load_matrix_condition/load_selfproj_condition vs. load_gate_condition.
"""
import torch


def participation_ratio_from_eigenvalues(eigenvalues: torch.Tensor) -> float:
    """PR = (sum lambda_i)^2 / sum(lambda_i^2), the formula itself. Takes raw eigenvalues rather
    than a matrix so it's trivially testable against hand-picked spectra (e.g. an all-equal
    spectrum of length d must give PR == d exactly) without needing to construct a matrix that
    HAS that spectrum first."""
    eigenvalues = eigenvalues.float()
    numerator = eigenvalues.sum() ** 2
    denominator = (eigenvalues ** 2).sum().clamp(min=1e-12)
    return (numerator / denominator).item()


def participation_ratio(conceptor: torch.Tensor) -> float:
    """conceptor: (d, d), the C matrix itself (compute_conceptor's output). C = R(R + alpha^-2 I)^-1
    is symmetric in exact arithmetic (R is symmetric, and R commutes with any function of itself,
    including (R + alpha^-2 I)^-1, so their product is symmetric) -- but matrix inversion can leave
    tiny floating-point asymmetry, so this defensively symmetrizes before eigh rather than assuming
    it. eigvalsh (not eigvals) is used deliberately: it's the numerically stable path for symmetric
    input and always returns real eigenvalues, matching test_steering_psr_conceptor.py's existing
    assumption that C's eigenvalues are real and in [0, 1)."""
    conceptor = conceptor.float()
    symmetrized = (conceptor + conceptor.T) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetrized)
    # Conceptor eigenvalues are mathematically in [0, 1) (see test_steering_psr_conceptor.py's
    # test_compute_conceptor_eigenvalues_bounded_in_unit_interval) but floating-point round-trip
    # through eigh can produce a value like -1e-7 for a near-zero eigenvalue -- clamp rather than
    # let a single slightly-negative term corrupt the sum-of-squares denominator.
    eigenvalues = eigenvalues.clamp(min=0.0)
    return participation_ratio_from_eigenvalues(eigenvalues)
