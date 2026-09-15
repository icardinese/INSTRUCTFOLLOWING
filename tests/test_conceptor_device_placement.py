"""Regression tests for a real crash: pool_separate_poles (steering/psr/data.py) deliberately
returns CPU tensors (saves GPU memory while accumulating across many training items), but nothing
downstream used to move them back to the live model's device -- so direction/conceptor/mu_instr/
delta_scale stayed on CPU while real hidden states were on CUDA, crashing with "Expected all
tensors to be on the same device, but found at least two devices, cuda:0 and cpu!" the moment a
hook tried to combine them.

Can't reproduce the actual cuda-vs-cpu mismatch in this CPU-only sandbox, so these tests verify
the FIX's mechanism directly: train_one_config must call `.to(device)` on whatever
load_or_pool_separate_poles returns, regardless of what device those tensors originally reported.
A spy on the mocked pool tensors' `.to()` method confirms the call actually happens with the
right target device -- this is exactly the call that was missing before the fix.
"""
import sys
from unittest.mock import MagicMock, patch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from tests.fakes import make_fake_model_and_tokenizer

ITEMS = [
    {"id": "0", "base_prompt": "explain this", "terse_prompt": "explain this tersely"},
    {"id": "1", "base_prompt": "what does it do", "terse_prompt": "what does it do, briefly"},
]
RESPONSES = {"0": "it adds two numbers", "1": "it returns a value"}


def _spy_pools(d=16, n=6):
    """Real CPU tensors, wrapped so calls to .to(...) are recorded but still actually move/copy
    the tensor (so downstream matrix math keeps working, not just the call-recording)."""
    real_base = torch.randn(n, d)
    real_instr = torch.randn(n, d)
    base_spy = MagicMock(wraps=real_base)
    instr_spy = MagicMock(wraps=real_instr)
    base_spy.to.side_effect = lambda *a, **k: real_base.to(*a, **k)
    instr_spy.to.side_effect = lambda *a, **k: real_instr.to(*a, **k)
    return base_spy, instr_spy


def test_conceptor_fixed_vector_moves_pooled_activations_to_device(tmp_path):
    from src.psr.conceptor.train import train_one_config

    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)
    base_spy, instr_spy = _spy_pools()

    with patch("src.psr.conceptor.train.load_or_pool_separate_poles", return_value=(base_spy, instr_spy)):
        train_one_config(
            model, tokenizer, layer_idx=1, alpha=4.0, seed=0, n_layers=4, device="cpu",
            train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
            cache_dir=tmp_path, n_epochs=1,
        )
    base_spy.to.assert_any_call("cpu")
    instr_spy.to.assert_any_call("cpu")


def test_conceptor_matrix_moves_pooled_activations_to_device(tmp_path):
    from src.psr.conceptor.matrix.train import train_one_config

    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)
    base_spy, instr_spy = _spy_pools()

    with patch("src.psr.conceptor.matrix.train.load_or_pool_separate_poles", return_value=(base_spy, instr_spy)):
        train_one_config(
            model, tokenizer, layer_idx=1, alpha=4.0, seed=0, n_layers=4, device="cpu",
            train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
            cache_dir=tmp_path, n_epochs=1,
        )
    base_spy.to.assert_any_call("cpu")
    instr_spy.to.assert_any_call("cpu")


def test_conceptor_selfproj_moves_pooled_activations_to_device(tmp_path):
    from src.psr.conceptor.selfproj.train import train_one_config

    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)
    base_spy, instr_spy = _spy_pools()

    with patch("src.psr.conceptor.selfproj.train.load_or_pool_separate_poles", return_value=(base_spy, instr_spy)):
        train_one_config(
            model, tokenizer, layer_idx=1, alpha=0.5, seed=0, n_layers=4, device="cpu",
            train_items=ITEMS, dev_items=ITEMS, train_responses=RESPONSES, dev_responses=RESPONSES,
            cache_dir=tmp_path, n_epochs=1,
        )
    base_spy.to.assert_any_call("cpu")
    instr_spy.to.assert_any_call("cpu")
