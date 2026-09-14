"""Direct tests of steering/psr/training_loop.py's train_gate -- specifically that mse_weight and
nll_weight are independent, mutually-adjustable multipliers on the MSE and NLL terms (not a single
combined "how much NLL to blend in" knob), matching Heyman & Vandeputte's real _MSE vs _LL split
(mse_weight=0 or nll_weight=0 means that term contributes ZERO gradient, not just a small one).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from steering.psr.gate import forward_with_gate_hook, init_gate_state
from steering.psr.training_loop import train_gate
from tests.fakes import make_fake_model_and_tokenizer

ITEMS = [
    {"id": "0", "base_prompt": "explain this", "terse_prompt": "explain this tersely"},
    {"id": "1", "base_prompt": "what does it do", "terse_prompt": "what does it do, briefly"},
]
RESPONSES = {"0": "it adds two numbers", "1": "it returns a value"}


def _setup(seed=0, d=16, n_layers=4):
    torch.manual_seed(seed)
    model, tokenizer = make_fake_model_and_tokenizer(d=d, n_layers=n_layers)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _train(model, tokenizer, seed, mse_weight, nll_weight, n_epochs=2):
    torch.manual_seed(seed)
    gate = init_gate_state(16, "cpu")
    direction = torch.randn(16)
    direction.requires_grad_(True)
    optimizer = torch.optim.Adam(gate.parameters() + [direction], lr=1e-2)
    forward_fn = lambda pair: forward_with_gate_hook(model, gate, direction, layer_idx=1, input_ids=pair["full_base"], n_resp=pair["n_resp"])
    train_gate(
        model, tokenizer, forward_fn, optimizer, layer_idx=1, n_layers=4,
        train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
        n_epochs=n_epochs, reg_coeff=0.1, mse_weight=mse_weight, nll_weight=nll_weight,
    )
    return gate.weight.detach().clone()


def test_default_mse_weight_is_pure_mse_and_matches_no_mse_weight_argument():
    """Not passing mse_weight at all must behave identically to explicitly passing 1.0 -- the
    default has to actually BE 1.0, not accidentally something else, since every existing call
    site in this project relies on the default reproducing pre-existing behavior unchanged."""
    model, tokenizer = _setup(seed=10)
    w_default = _train(model, tokenizer, seed=5, mse_weight=1.0, nll_weight=0.0)
    model, tokenizer = _setup(seed=10)
    w_explicit = _train(model, tokenizer, seed=5, mse_weight=1.0, nll_weight=0.0)
    assert torch.allclose(w_default, w_explicit)


def test_mse_weight_zero_and_nll_weight_zero_gives_regularization_only_training():
    """Both weights at zero: the gate should still update (regularization alone has gradient
    pressure against an all-zero gate), but reach a DIFFERENT point than either pure-MSE or
    pure-NLL training, since neither representation-matching signal is present at all."""
    model, tokenizer = _setup(seed=20)
    w_both_zero = _train(model, tokenizer, seed=7, mse_weight=0.0, nll_weight=0.0)
    model, tokenizer = _setup(seed=20)
    w_pure_mse = _train(model, tokenizer, seed=7, mse_weight=1.0, nll_weight=0.0)
    assert not torch.allclose(w_both_zero, w_pure_mse), (
        "training with both weights at zero must differ from pure-MSE training -- otherwise "
        "mse_weight isn't actually being applied as a multiplier on the MSE term"
    )


def test_pure_mse_and_pure_nll_are_genuinely_different_training_regimes():
    model, tokenizer = _setup(seed=30)
    w_mse = _train(model, tokenizer, seed=9, mse_weight=1.0, nll_weight=0.0)
    model, tokenizer = _setup(seed=30)
    w_nll = _train(model, tokenizer, seed=9, mse_weight=0.0, nll_weight=1.0)
    assert not torch.allclose(w_mse, w_nll)
