"""Tests for steering/psr/gate.py -- the shared chassis every PSR variant depends on. If this
breaks, every variant breaks, so this is the highest-value file in the whole test suite.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from steering.psr.gate import (
    GateState,
    answer_only_mask,
    coefficient,
    forward_with_gate_hook,
    init_gate_state,
    location_fit,
    make_inference_hook,
    regularization_loss,
    subsequent_layers_mse,
)
from tests.fakes import make_fake_model_and_tokenizer


def test_answer_only_mask_shape_and_values():
    mask = answer_only_mask(seq_len=6, n_resp=3, device="cpu")
    assert mask.tolist() == [[[False], [False], [False], [True], [True], [True]]]


def test_location_fit_matches_manual_relu_computation():
    torch.manual_seed(0)
    gate = init_gate_state(hidden_size=16, device="cpu")
    hidden = torch.randn(1, 5, 16)
    fit = location_fit(gate, hidden)
    expected = torch.relu(hidden.float() @ gate.weight + gate.bias)
    assert torch.allclose(fit, expected)


def test_coefficient_masking_zeroes_prompt_span():
    gate = GateState(weight=torch.ones(16, 1) * 0.5, bias=torch.tensor([0.1]), coeff_bias=torch.tensor([0.0]))
    hidden = torch.ones(1, 6, 16)
    mask = answer_only_mask(6, 3, "cpu")
    coeff, fit = coefficient(gate, hidden, mask)
    assert torch.all(fit[:, :3, :] == 0), "prompt-span positions should be hard-zeroed by the mask"
    assert torch.all(fit[:, 3:, :] > 0), "response-span positions should be nonzero"


def test_regularization_penalizes_all_zero_gate():
    fit_all_zero = torch.zeros(1, 5, 1)
    fit_saturated = torch.ones(1, 5, 1) * 2
    reg_zero = regularization_loss(fit_all_zero, reg_coeff=0.1)
    reg_saturated = regularization_loss(fit_saturated, reg_coeff=0.1)
    assert reg_zero.item() > reg_saturated.item(), "an all-zero gate should be penalized more than a firing one"


def test_forward_with_gate_hook_gradient_flows_into_gate_only():
    model, _ = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)
    gate = init_gate_state(16, "cpu")
    direction = torch.randn(16)
    input_ids = torch.randint(0, 200, (1, 6))

    hidden_pred, fit = forward_with_gate_hook(model, gate, direction, layer_idx=1, input_ids=input_ids, n_resp=3)
    with torch.no_grad():
        target = model(input_ids).hidden_states
    mse = subsequent_layers_mse(hidden_pred, target, layer_idx=1, n_resp=3, n_layers=4)
    loss = mse + regularization_loss(fit, 0.1)
    loss.backward()

    for name, p in zip(["weight", "bias", "coeff_bias"], gate.parameters()):
        assert p.grad is not None, f"{name} should have received a gradient"
    for p in model.parameters():
        assert p.grad is None, "frozen base model params should NOT accumulate gradient"


def test_make_inference_hook_prefill_is_noop_generation_step_is_steered():
    gate = GateState(weight=torch.ones(16, 1) * 0.5, bias=torch.tensor([0.1]), coeff_bias=torch.tensor([0.0]))
    direction = torch.ones(16)
    hook = make_inference_hook(gate, direction)

    prefill = torch.randn(1, 5, 16)
    assert torch.equal(hook(prefill), prefill), "prefill pass (seq_len > 1) must be a no-op"

    gen_step = torch.ones(1, 1, 16)
    out = hook(gen_step)
    assert not torch.equal(out, gen_step), "a single generation step must be steered"
    expected_coeff = torch.relu(torch.tensor(16 * 0.5 + 0.1))
    expected = gen_step + expected_coeff * direction
    assert torch.allclose(out, expected, atol=1e-5)
