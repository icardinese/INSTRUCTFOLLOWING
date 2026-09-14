"""Genuine matrix application: correction(h) = coeff(h) * (C @ (mu_instr - h)) / scale, evaluated
fresh at every position -- unlike steering/psr/conceptor/direction.py's project_direction, which
collapses C to a single fixed vector once and never touches C again. Here C filters the ACTUAL gap
between the current hidden state and the instructed-mean target through the concept's subspace.

Named `gate_correction`/`forward_with_gate_hook` identically to steering/psr/gate.py's versions --
disambiguated by import path (steering.psr.conceptor.matrix.logic vs steering.psr.gate), not by
suffix. Same short name, same job, genuinely different formula.
"""
import torch

from steering.psr.gate import answer_only_mask, coefficient, GateState


def compute_delta_scale(conceptor: torch.Tensor, mu_instr: torch.Tensor, reference_pool: torch.Tensor) -> torch.Tensor:
    """Typical norm of C @ (mu_instr - h) over real activations. Needed because raw residual-stream
    magnitudes aren't O(1) -- without this, an untrained gate's near-zero coefficient times an
    UNNORMALIZED C @ delta still actively corrupts the hidden state (confirmed empirically: an
    untrained baseline measured MSE=93 vs ~33 for every properly-scaled method)."""
    delta = mu_instr.unsqueeze(0) - reference_pool
    c_delta = delta @ conceptor
    return c_delta.norm(dim=-1).mean().clamp(min=1e-6)


def gate_correction(gate: GateState, conceptor: torch.Tensor, mu_instr: torch.Tensor, delta_scale: torch.Tensor, hidden: torch.Tensor, mask=None):
    coeff, fit = coefficient(gate, hidden, mask)
    delta = mu_instr.to(hidden.dtype) - hidden
    c_delta = (delta @ conceptor.to(hidden.dtype)) / delta_scale.to(hidden.dtype)
    return (coeff * c_delta).to(hidden.dtype), fit


def forward_with_gate_hook(model, gate: GateState, conceptor, mu_instr, delta_scale, layer_idx: int, input_ids: torch.Tensor, n_resp: int):
    """Returns (hidden_states, logits, fit_vals) -- see steering/psr/gate.py's version of this
    function for why logits are returned now (steering/psr/nll.py's auxiliary loss)."""
    from core.model_common import get_decoder_layers
    layer = get_decoder_layers(model)[layer_idx]
    captured_fit = {}

    def wrapped(module, inputs, output):
        hidden = output[0]
        mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
        correction, fit = gate_correction(gate, conceptor, mu_instr, delta_scale, hidden, mask)
        captured_fit["fit"] = fit
        return (hidden + correction,) + tuple(output[1:])

    handle = layer.register_forward_hook(wrapped)
    try:
        out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        handle.remove()
    return out.hidden_states, out.logits, captured_fit["fit"]


def make_inference_hook(gate: GateState, conceptor, mu_instr, delta_scale):
    """Inference-time hook for use with steering.hooks.steering_hook -- same prefill-vs-decode
    split as steering/psr/gate.py's make_inference_hook (prefill = full prompt, don't steer;
    single generated token = the response by construction, always steer)."""
    @torch.no_grad()
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            return hidden
        correction, _ = gate_correction(gate, conceptor, mu_instr, delta_scale, hidden, mask=None)
        return hidden + correction
    return hook_fn
