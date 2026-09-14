"""End-to-end (fake model, no real weights) test of src/psr/proper/train.py's train_one_config --
exercises the full path: build_training_pair -> forward_with_gate_hook (now returning logits too)
-> train_gate's combined MSE + regularization + NLL loss -> a real backward/optimizer step. This
is the integration point every other unit test in this project stops short of; it exists because
the NLL-loss wiring touches five files at once (gate.py, training_loop.py, nll.py, and every
variant's train.py) and a passing unit test in each individually wouldn't catch a mismatched
tuple-unpacking bug at the seams between them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.psr.proper.train import train_one_config
from tests.fakes import make_fake_model_and_tokenizer

ITEMS = [
    {"id": "0", "base_prompt": "explain this", "terse_prompt": "explain this tersely"},
    {"id": "1", "base_prompt": "what does it do", "terse_prompt": "what does it do, briefly"},
]
RESPONSES = {"0": "it adds two numbers", "1": "it returns a value"}


def _setup(d=16, n_layers=4):
    model, tokenizer = make_fake_model_and_tokenizer(d=d, n_layers=n_layers)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def test_train_one_config_runs_and_reduces_baseline_vs_untrained_gate():
    torch.manual_seed(0)
    model, tokenizer = _setup()
    result = train_one_config(
        model, tokenizer, layer_idx=1, seed=0, n_layers=4, hidden_size=16, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        n_epochs=2, nll_weight=0.0,
    )
    for key in ["baseline_mse", "baseline_nll", "final_mse", "final_nll", "weight", "bias", "coeff_bias", "direction"]:
        assert key in result
    assert result["final_mse"] >= 0.0
    assert result["final_nll"] >= 0.0
    # Training minimizes MSE + reg (+ nll_weight * nll); with nll_weight=0 the MSE term alone is
    # what's being optimized, so final MSE should be no worse than doing nothing (baseline).
    assert result["final_mse"] <= result["baseline_mse"] + 1e-3


def test_nonzero_nll_weight_changes_trained_gate_versus_zero_weight():
    """A sanity check that nll_weight actually participates in the gradient (not just computed and
    discarded) -- training the SAME (seeded) starting point with nll_weight=0 vs. nll_weight=1.0
    should converge to measurably different gate weights."""
    model, tokenizer = _setup()

    result_zero = train_one_config(
        model, tokenizer, layer_idx=1, seed=1, n_layers=4, hidden_size=16, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        n_epochs=2, nll_weight=0.0,
    )
    result_weighted = train_one_config(
        model, tokenizer, layer_idx=1, seed=1, n_layers=4, hidden_size=16, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        n_epochs=2, nll_weight=1.0,
    )
    assert not torch.allclose(result_zero["weight"], result_weighted["weight"]), (
        "nll_weight=1.0 vs 0.0 must produce different trained gate weights from the same seed -- "
        "if they're identical, the auxiliary NLL loss isn't actually contributing gradient."
    )


def test_pure_nll_training_mse_weight_zero_differs_from_pure_mse():
    """mse_weight=0.0 must mean the MSE term contributes NO gradient at all -- a genuine,
    mutually-exclusive alternative (matching Heyman & Vandeputte's actual _MSE vs _LL split),
    not just a smaller nudge. Training pure-MSE (mse_weight=1, nll_weight=0) vs pure-NLL
    (mse_weight=0, nll_weight=1) from the same seed must diverge."""
    model, tokenizer = _setup()

    pure_mse = train_one_config(
        model, tokenizer, layer_idx=1, seed=3, n_layers=4, hidden_size=16, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        n_epochs=2, mse_weight=1.0, nll_weight=0.0,
    )
    pure_nll = train_one_config(
        model, tokenizer, layer_idx=1, seed=3, n_layers=4, hidden_size=16, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        n_epochs=2, mse_weight=0.0, nll_weight=1.0,
    )
    assert not torch.allclose(pure_mse["weight"], pure_nll["weight"]), (
        "pure MSE and pure NLL training from the same seed must diverge -- if identical, "
        "mse_weight isn't actually gating the MSE term's contribution to the gradient."
    )
    # Both still REPORT both metrics (diagnostics are always computed regardless of weights).
    for result in (pure_mse, pure_nll):
        assert result["final_mse"] >= 0.0
        assert result["final_nll"] >= 0.0


def test_on_epoch_end_receives_live_gate_and_direction_each_epoch():
    model, tokenizer = _setup()
    seen_epochs = []

    def on_epoch_end(epoch, gate, direction, dev_metrics):
        seen_epochs.append(epoch)
        assert gate.weight.shape == (16, 1)
        assert direction.shape == (16,)
        assert "mse" in dev_metrics and "nll" in dev_metrics

    train_one_config(
        model, tokenizer, layer_idx=1, seed=2, n_layers=4, hidden_size=16, device="cpu",
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        n_epochs=3, on_epoch_end=on_epoch_end,
    )
    assert seen_epochs == [0, 1, 2]
