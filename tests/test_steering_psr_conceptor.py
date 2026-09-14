"""Tests for steering/psr/conceptor/* -- the closed-form conceptor math and both correction
variants. Includes a regression test for the delta_scale bug: an untrained gate must produce a
SMALL correction at realistic activation scale, not the ~93 MSE blowup that was caught and fixed
mid-session when delta_scale normalization was missing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from steering.psr.conceptor.direction import compute_conceptor, project_direction
from steering.psr.conceptor.matrix.logic import compute_delta_scale as matrix_delta_scale
from steering.psr.conceptor.matrix.logic import gate_correction as matrix_gate_correction
from steering.psr.conceptor.rank_diagnostic import participation_ratio, participation_ratio_from_eigenvalues
from steering.psr.gate import init_gate_state


def test_compute_conceptor_eigenvalues_bounded_in_unit_interval():
    """C = R(R + alpha^-2 I)^-1 must have eigenvalues in [0, 1) -- this is the actual reason
    conceptors are more stable than an arbitrary learned matrix: they can only attenuate, never
    amplify, any direction."""
    torch.manual_seed(0)
    activations = torch.randn(50, 8) * 10
    C = compute_conceptor(activations, alpha=4.0)
    eigvals = torch.linalg.eigvalsh(C)
    assert torch.all(eigvals >= -1e-5), "conceptor eigenvalues must be non-negative"
    assert torch.all(eigvals < 1.0 + 1e-5), "conceptor eigenvalues must be strictly less than 1"


def test_project_direction_returns_unit_vector():
    torch.manual_seed(1)
    activations = torch.randn(50, 8) * 10
    C = compute_conceptor(activations, alpha=4.0)
    base_direction = torch.randn(8)
    base_direction = base_direction / base_direction.norm()
    projected = project_direction(C, base_direction)
    assert abs(projected.norm().item() - 1.0) < 1e-5


def test_matrix_variant_untrained_gate_gives_small_correction_at_realistic_scale():
    """Regression test for the real bug caught this session: without delta_scale normalization,
    an untrained gate's tiny coefficient times an UNNORMALIZED C @ delta gave a correction with
    norm ~90+ on realistic-scale activations (measured baseline_dev_mse=93 vs ~33 everywhere else).
    """
    torch.manual_seed(2)
    pool = torch.randn(50, 16) * 15  # realistic residual-stream scale, not unit-norm toy data
    C = compute_conceptor(pool, alpha=4.0)
    mu_instr = torch.randn(16) * 15
    delta_scale = matrix_delta_scale(C, mu_instr, pool)

    gate = init_gate_state(16, "cpu")  # untrained, tiny random weight
    hidden = torch.randn(1, 1, 16) * 15
    correction, _ = matrix_gate_correction(gate, C, mu_instr, delta_scale, hidden)
    assert correction.norm().item() < 5.0, (
        f"untrained gate should give a small correction, got norm={correction.norm().item():.2f} "
        f"-- this is the exact bug that produced baseline_dev_mse=93 earlier this session"
    )


def test_selfproj_delta_shrinks_when_h_aligns_with_dominant_direction():
    """A hidden state already aligned with C's dominant eigenvector should have a smaller
    (C @ h - h) delta than a random-direction one of the same magnitude -- this is the theoretical
    property the self-projection design is supposed to have. Tests the delta directly, not through
    a gate's coefficient -- an untrained gate's own random init can independently land at a
    zero-firing coefficient for a specific draw (a separate, already-known flakiness class from
    make_inference_hook's test), which would confound this comparison if routed through gate_correction."""
    torch.manual_seed(3)
    pool = torch.randn(200, 16) * 10
    C = compute_conceptor(pool, alpha=4.0)

    eigvals, eigvecs = torch.linalg.eigh((pool.T @ pool) / pool.shape[0])
    top_eigvec = eigvecs[:, torch.argmax(eigvals)]

    h_aligned = (top_eigvec * 30.0).view(1, 1, 16)
    h_random = torch.randn(1, 1, 16) * 30.0

    delta_aligned = (h_aligned @ C - h_aligned).norm()
    delta_random = (h_random @ C - h_random).norm()
    assert delta_aligned.item() < delta_random.item()


def test_participation_ratio_of_uniform_spectrum_equals_dimensionality():
    """PR = (sum lambda)^2 / sum(lambda^2). For d equal eigenvalues this simplifies exactly to d --
    the "spread evenly across every dimension" end of the scale."""
    eigenvalues = torch.ones(8) * 0.37  # value shouldn't matter, only that they're all equal
    pr = participation_ratio_from_eigenvalues(eigenvalues)
    assert abs(pr - 8.0) < 1e-4


def test_participation_ratio_of_rank_one_spectrum_equals_one():
    """A single nonzero eigenvalue (everything else exactly zero) is the "acts like one direction"
    end of the scale -- PR must come out to exactly 1, matching plain PSR's rank-1 correction."""
    eigenvalues = torch.tensor([5.0, 0.0, 0.0, 0.0])
    pr = participation_ratio_from_eigenvalues(eigenvalues)
    assert abs(pr - 1.0) < 1e-4


def test_participation_ratio_of_real_conceptor_is_between_one_and_dimensionality():
    """End-to-end: build a real conceptor from real (non-degenerate) data and confirm its PR lands
    in the theoretically valid range (1, d] rather than trusting the two synthetic-spectrum tests
    alone to catch a wiring mistake in participation_ratio()'s matrix -> eigenvalues path."""
    torch.manual_seed(5)
    activations = torch.randn(100, 16) * 10
    C = compute_conceptor(activations, alpha=4.0)
    pr = participation_ratio(C)
    assert 1.0 <= pr <= 16.0


def test_participation_ratio_increases_as_aperture_widens():
    """A larger alpha lets C attenuate less (closer to identity, see compute_conceptor's docstring),
    which should spread its eigenvalues more evenly and raise PR -- this is the actual empirical
    hook the project wants (\"does dimensionality increase or decrease as alpha/aperture changes\"),
    checked in the one direction that's unambiguous from the math: alpha -> very large means C -> I,
    whose PR is exactly d, the maximum possible -- so PR must be monotonically approaching it here."""
    torch.manual_seed(6)
    activations = torch.randn(100, 16) * 10
    pr_tight = participation_ratio(compute_conceptor(activations, alpha=0.5))
    pr_loose = participation_ratio(compute_conceptor(activations, alpha=50.0))
    assert pr_loose > pr_tight
