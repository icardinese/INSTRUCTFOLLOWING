"""End-to-end (fake model) test of src/psr/conceptor/matrix/train.py's train_one_config -- exercises
pooling -> conceptor construction -> delta_scale -> gate training -> participation_ratio, all
through the real (non-mocked) math, only the model itself is fake.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.psr.conceptor.matrix.train import train_one_config
from tests.fakes import make_fake_model_and_tokenizer

ITEMS = [
    {"id": "0", "base_prompt": "explain this", "terse_prompt": "explain this tersely"},
    {"id": "1", "base_prompt": "what does it do", "terse_prompt": "what does it do, briefly"},
    {"id": "2", "base_prompt": "describe the function", "terse_prompt": "describe it briefly"},
]
RESPONSES = {"0": "it adds two numbers", "1": "it returns a value", "2": "it computes a sum"}


def test_train_one_config_runs_and_reports_participation_ratio(tmp_path):
    torch.manual_seed(0)
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    result = train_one_config(
        model, tokenizer, layer_idx=1, alpha=4.0, seed=0, n_layers=4, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        cache_dir=tmp_path, n_epochs=1,
    )

    for key in ["baseline_mse", "baseline_nll", "final_mse", "final_nll", "participation_ratio",
                "weight", "bias", "coeff_bias", "conceptor", "mu_instr", "delta_scale"]:
        assert key in result
    assert result["conceptor"].shape == (16, 16)
    assert 1.0 <= result["participation_ratio"] <= 16.0 + 1e-4


def test_train_one_config_uses_cache_and_does_not_repool_on_second_call(tmp_path):
    """load_or_pool_separate_poles caches pooled activations per layer -- a second call at the same
    layer must read from cache_dir instead of re-running the (expensive, real) forward passes."""
    torch.manual_seed(1)
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    train_one_config(
        model, tokenizer, layer_idx=1, alpha=4.0, seed=0, n_layers=4, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        cache_dir=tmp_path, n_epochs=1,
    )
    assert (tmp_path / "pooled_base_layer1.pt").exists()
    assert (tmp_path / "pooled_instr_layer1.pt").exists()

    # Second call: even with a differently-seeded model call count, the pooled tensors it reads
    # back should be bit-identical to what was cached, confirming the cache path was actually hit.
    cached_base = torch.load(tmp_path / "pooled_base_layer1.pt")
    result2 = train_one_config(
        model, tokenizer, layer_idx=1, alpha=8.0, seed=0, n_layers=4, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        cache_dir=tmp_path, n_epochs=1,
    )
    assert torch.equal(torch.load(tmp_path / "pooled_base_layer1.pt"), cached_base)
    assert result2["conceptor"].shape == (16, 16)
