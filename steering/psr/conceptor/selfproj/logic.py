"""Self-projection: correction(h) = coeff(h) * (C @ h - h) / scale. No mu_instr, no population-mean
target -- pushes h toward ITS OWN projection onto the concept-relevant subspace, using only
information already in h. Closer to the conceptor literature's standard "soft filter" usage than
the matrix/ variant's external-target version.
"""
import torch

from steering.psr.gate import answer_only_mask, coefficient, GateState


def compute_delta_scale(conceptor: torch.Tensor, reference_pool: torch.Tensor) -> torch.Tensor:
    """Typical norm of (C @ h - h) over real activations. Same normalization need as
    conceptor/matrix/'s compute_delta_scale, different quantity being normalized -- watch the
    aperture here specifically: if C is close to the identity matrix (alpha too loose relative to
    the real eigenvalue scale), this comes out near-zero and the correction has nothing real to
    learn from. See steering/psr/conceptor/rank_diagnostic.py before trusting a given alpha here."""
    c_h = reference_pool @ conceptor
    delta = c_h - reference_pool
    return delta.norm(dim=-1).mean().clamp(min=1e-6)


def gate_correction(gate: GateState, conceptor: torch.Tensor, delta_scale: torch.Tensor, hidden: torch.Tensor, mask=None):
    coeff, fit = coefficient(gate, hidden, mask)
    c_h = hidden.float() @ conceptor.to(hidden.dtype).float()
    delta = (c_h - hidden.float()) / delta_scale.to(hidden.dtype)
    return (coeff * delta).to(hidden.dtype), fit


def forward_with_gate_hook(model, gate: GateState, conceptor, delta_scale, layer_idx: int, input_ids: torch.Tensor, n_resp: int):
    """Returns (hidden_states, logits, fit_vals) -- see steering/psr/gate.py's version of this
    function for why logits are returned now (steering/psr/nll.py's auxiliary loss)."""
    from core.model_common import get_decoder_layers
    layer = get_decoder_layers(model)[layer_idx]
    captured_fit = {}

    def wrapped(module, inputs, output):
        hidden = output[0]
        mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
        correction, fit = gate_correction(gate, conceptor, delta_scale, hidden, mask)
        captured_fit["fit"] = fit
        return (hidden + correction,) + tuple(output[1:])

    handle = layer.register_forward_hook(wrapped)
    try:
        out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        handle.remove()
    return out.hidden_states, out.logits, captured_fit["fit"]


def make_inference_hook(gate: GateState, conceptor, delta_scale):
    """Inference-time hook, same prefill-vs-decode split as every other variant's."""
    @torch.no_grad()
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            return hidden
        correction, _ = gate_correction(gate, conceptor, delta_scale, hidden, mask=None)
        return hidden + correction
    return hook_fn
