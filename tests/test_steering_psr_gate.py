"""Tests for steering/psr/gate.py -- the shared chassis every PSR variant depends on. If this
breaks, every variant breaks, so this is the highest-value file in the whole test suite.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
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
from steering.psr.nll import response_nll
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

    hidden_pred, logits, fit = forward_with_gate_hook(model, gate, direction, layer_idx=1, input_ids=input_ids, n_resp=3)
    assert logits.shape == (1, 6, 200), "logits should be (batch, seq_len, vocab) from the same corrected forward pass"
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


def test_forward_with_gate_hook_works_when_decoder_layer_returns_a_plain_tensor():
    """Regression test for a REAL crash: some transformers versions return a decoder layer's
    hidden_states as a plain tensor, not a (hidden_states, ...) tuple. A hook that assumes
    "always a tuple" (output[0]) silently mis-indexes a plain tensor instead of raising, then
    wraps the result BACK into a tuple -- so the NEXT layer receives a tuple where it expects a
    tensor, crashing several frames downstream with `AttributeError: 'tuple' object has no
    attribute 'dtype'`. This is exactly why TinyLayer/TinyModel gained a layer_returns_tuple=False
    mode: the previous fake model only ever returned tuples, so nothing in this suite could have
    caught this until now."""
    model, _ = make_fake_model_and_tokenizer(d=16, n_layers=4, layer_returns_tuple=False)
    for p in model.parameters():
        p.requires_grad_(False)
    gate = init_gate_state(16, "cpu")
    direction = torch.randn(16)
    input_ids = torch.randint(0, 200, (1, 6))

    # forward_with_gate_hook hooks layer 1 of 4 -- if the hook mishandles the plain-tensor output,
    # layers 2 and 3 receive a corrupted (tuple-wrapped) hidden_states and this raises deep inside
    # TinyModel's own forward loop (`h = out[0] if isinstance(out, tuple) else out` would then be
    # operating on an already-wrong value, or a downstream shape/type mismatch would surface).
    hidden_pred, logits, fit = forward_with_gate_hook(model, gate, direction, layer_idx=1, input_ids=input_ids, n_resp=3)
    assert logits.shape == (1, 6, 200)
    assert hidden_pred[-1].shape == (1, 6, 16), "final hidden state must still be a real tensor of the right shape, not a mis-wrapped tuple"


def test_forward_with_gate_hook_gives_the_same_correction_regardless_of_layer_output_shape():
    """The steering correction itself must be identical whether the underlying decoder layer
    returns a tuple or a plain tensor -- these are two different transformers conventions for the
    SAME semantic output, so the correction shouldn't depend on which one is in use."""
    torch.manual_seed(11)
    model_tuple, _ = make_fake_model_and_tokenizer(d=16, n_layers=4, layer_returns_tuple=True)
    torch.manual_seed(11)
    model_plain, _ = make_fake_model_and_tokenizer(d=16, n_layers=4, layer_returns_tuple=False)
    for m in (model_tuple, model_plain):
        for p in m.parameters():
            p.requires_grad_(False)

    gate = init_gate_state(16, "cpu")
    direction = torch.randn(16)
    input_ids = torch.randint(0, 200, (1, 6))

    hidden_tuple, logits_tuple, _ = forward_with_gate_hook(model_tuple, gate, direction, layer_idx=1, input_ids=input_ids, n_resp=3)
    hidden_plain, logits_plain, _ = forward_with_gate_hook(model_plain, gate, direction, layer_idx=1, input_ids=input_ids, n_resp=3)
    assert torch.allclose(hidden_tuple[-1], hidden_plain[-1], atol=1e-5)
    assert torch.allclose(logits_tuple, logits_plain, atol=1e-5)


def test_response_nll_matches_manual_shifted_cross_entropy():
    """Directly checks the equation (Eq. 4: -sum_t log P(y_t | y_<t, x; h'_x)) against a manual,
    unshifted computation on tiny synthetic logits -- no fake model needed, this is pure tensor math."""
    torch.manual_seed(4)
    vocab, seq_len, n_resp = 10, 6, 3
    logits = torch.randn(1, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (1, seq_len))

    got = response_nll(logits, input_ids, n_resp, reduction="sum")

    expected = torch.zeros(())
    for t in range(seq_len - n_resp, seq_len):
        log_probs = torch.log_softmax(logits[0, t - 1], dim=-1)
        expected = expected - log_probs[input_ids[0, t]]
    assert torch.allclose(got, expected, atol=1e-5)


def test_response_nll_lower_for_confident_correct_predictions():
    """A degenerate logit distribution that puts all mass on the actual target token at every
    response position should give a near-zero NLL; a uniform distribution should give a much
    larger one (~log(vocab) per token) -- sanity check that the loss actually measures what it
    claims to, not just that the shapes line up."""
    vocab, seq_len, n_resp = 20, 5, 2
    input_ids = torch.tensor([[3, 7, 1, 9, 15]])

    confident_logits = torch.full((1, seq_len, vocab), -10.0)
    for t in range(seq_len - n_resp, seq_len):
        confident_logits[0, t - 1, input_ids[0, t]] = 10.0
    uniform_logits = torch.zeros(1, seq_len, vocab)

    confident_nll = response_nll(confident_logits, input_ids, n_resp, reduction="sum")
    uniform_nll = response_nll(uniform_logits, input_ids, n_resp, reduction="sum")
    assert confident_nll.item() < 1e-3
    assert uniform_nll.item() > n_resp * (torch.log(torch.tensor(float(vocab))).item() - 0.01)


def test_response_nll_rejects_n_resp_larger_than_available_context():
    logits = torch.randn(1, 4, 10)
    input_ids = torch.randint(0, 10, (1, 4))
    with pytest.raises(ValueError):
        response_nll(logits, input_ids, n_resp=4, reduction="sum")
