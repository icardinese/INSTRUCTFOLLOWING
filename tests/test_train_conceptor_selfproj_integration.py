"""End-to-end (fake model) test of src/psr/conceptor/selfproj/train.py's train_one_config -- covers
both the normal-training path and the "delta_scale too small, skip" path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.psr.conceptor.selfproj.train import adaptive_alpha_grid, eigendecompose, train_one_config
from tests.fakes import make_fake_model_and_tokenizer

ITEMS = [
    {"id": "0", "base_prompt": "explain this", "terse_prompt": "explain this tersely"},
    {"id": "1", "base_prompt": "what does it do", "terse_prompt": "what does it do, briefly"},
    {"id": "2", "base_prompt": "describe the function", "terse_prompt": "describe it briefly"},
]
RESPONSES = {"0": "it adds two numbers", "1": "it returns a value", "2": "it computes a sum"}


def test_train_one_config_trains_and_reports_participation_ratio_at_tight_alpha(tmp_path):
    torch.manual_seed(0)
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    # A small alpha (tight aperture) keeps C well away from identity, so delta_scale shouldn't
    # collapse to ~0 for this test -- exercises the "normal" (non-skipped) path.
    result = train_one_config(
        model, tokenizer, layer_idx=1, alpha=0.5, seed=0, n_layers=4, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        cache_dir=tmp_path, n_epochs=1,
    )
    assert result["skipped"] is False
    for key in ["baseline_mse", "final_mse", "participation_ratio", "weight", "conceptor"]:
        assert key in result


def test_train_one_config_skips_when_delta_scale_is_too_small(tmp_path, monkeypatch):
    """Forces the skip branch deterministically (via monkeypatch) rather than relying on an
    extreme alpha value, whose delta_scale sits right at float32's precision floor with real
    (small-sample) fake-model activations and was observed to be numerically borderline/flaky
    across BLAS thread counts -- see the actual math property (delta shrinks as C -> identity)
    already covered by test_steering_psr_conceptor.py's synthetic, well-conditioned-data tests."""
    import src.psr.conceptor.selfproj.train as selfproj_train

    monkeypatch.setattr(selfproj_train, "compute_delta_scale", lambda conceptor, pool: torch.tensor(1e-6))

    torch.manual_seed(2)
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    result = train_one_config(
        model, tokenizer, layer_idx=1, alpha=4.0, seed=0, n_layers=4, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        cache_dir=tmp_path, n_epochs=1,
    )
    assert result["skipped"] is True
    assert "participation_ratio" in result
    assert "weight" not in result, "a skipped point must not claim to have trained anything"


def test_adaptive_alpha_grid_spans_from_tight_to_loose():
    """Percentile here follows the ORIGINAL (pre-existing) convention: idx = (100-p)/100 * d, so a
    HIGHER percentile p looks at a LARGER eigenvalue (smaller index into the descending-sorted
    spectrum) -> a SMALLER alpha. Confirmed against the actual function rather than an assumed
    direction, since this convention was inherited as-is from the pre-refactor script."""
    torch.manual_seed(3)
    pool = torch.randn(50, 16) * 10
    eigvals, _, _ = eigendecompose(pool)
    grid = adaptive_alpha_grid(eigvals, percentiles=[10, 90])
    alphas = [a for _, a in grid]
    assert alphas[0] > alphas[1], "p=10 looks at a smaller eigenvalue than p=90, so its alpha (1/sqrt(eig)) must be larger"
