"""The token-gating mechanism shared by every PSR variant: a ReLU probe producing a per-position
coefficient, masked to the response span, regularized against collapsing to always-zero. This is
Heyman & Vandeputte's actual PSR design. What varies BETWEEN variants (proper/conceptor/matrix/
selfproj) is only where `direction` comes from and how it combines with the gate's coefficient --
none of that lives here, see steering/psr/proper/, steering/psr/conceptor/, etc.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core.model_common import get_decoder_layers


@dataclass
class GateState:
    """The trainable chassis every PSR variant shares. Deliberately does NOT include `direction` --
    direction is sometimes a trainable leaf tensor (psr/proper) and sometimes a fixed, precomputed
    one (psr/conceptor/*); bundling it here would force every variant to pretend those are the same
    kind of thing. Callers needing a trainable direction just add it to the optimizer's parameter
    list alongside gate.parameters(), not inside this dataclass."""
    weight: torch.Tensor       # (hidden_size, 1)
    bias: torch.Tensor         # (1,)
    coeff_bias: torch.Tensor   # (1,) -- learned additive offset on the coefficient

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight, self.bias, self.coeff_bias]


def init_gate_state(hidden_size: int, device: str, dtype: torch.dtype = torch.float32) -> GateState:
    weight = (torch.randn(hidden_size, 1, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
    coeff_bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
    return GateState(weight=weight, bias=bias, coeff_bias=coeff_bias)


def location_fit(gate: GateState, hidden: torch.Tensor) -> torch.Tensor:
    """How much THIS position needs steering. (batch, seq, d) -> (batch, seq, 1)."""
    return torch.relu(hidden.float() @ gate.weight + gate.bias)


def answer_only_mask(seq_len: int, n_resp: int, device) -> torch.Tensor:
    """True for the last n_resp positions (the response span). PSR's whole premise is selective
    intervention, so this is enforced directly rather than left for the gate to learn on its own."""
    positions = torch.arange(seq_len, device=device)
    return (positions >= (seq_len - n_resp)).view(1, seq_len, 1)


def coefficient(gate: GateState, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (coeff, fit). coeff is what every variant's correction gets scaled by; fit is exposed
    separately since regularization_loss needs it directly, not the scaled coefficient."""
    fit = location_fit(gate, hidden)
    if mask is not None:
        fit = torch.where(mask, fit, torch.zeros_like(fit))
    desired_presence = 1.0 + gate.coeff_bias
    return desired_presence * fit, fit


def regularization_loss(fit_vals: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    """Penalizes the gate for firing NOWHERE across a whole response -- the dead-ReLU guard. Without
    this, relu(1 - sum(fit)) < 0 gives zero gradient pressure away from an all-zero (do-nothing) gate."""
    fit_sum = fit_vals.sum(dim=1)
    return reg_coeff * torch.relu(1.0 - fit_sum).mean()


def forward_with_gate_hook(model, gate: GateState, direction: torch.Tensor, layer_idx: int, input_ids: torch.Tensor, n_resp: int):
    """Live, gradient-tracked forward pass with the gate's correction active at layer_idx. Returns
    (hidden_states tuple, logits, fit_vals). Routes through get_decoder_layers rather than a
    hardcoded model.model.layers[layer_idx] -- one line change that makes every PSR variant
    automatically architecture-portable once IFEval models come into play.

    logits come from this SAME forward pass (the model call below already computes them for a
    causal LM head) -- returning them costs nothing extra and is what steering/psr/nll.py's
    auxiliary NLL loss needs; before this, they were silently discarded via `out.hidden_states`
    alone."""
    layer = get_decoder_layers(model)[layer_idx]
    captured_fit = {}

    def wrapped(module, inputs, output):
        hidden = output[0]
        mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
        coeff, fit = coefficient(gate, hidden, mask)
        correction = (coeff * direction.to(hidden.dtype)).to(hidden.dtype)
        captured_fit["fit"] = fit
        return (hidden + correction,) + tuple(output[1:])

    handle = layer.register_forward_hook(wrapped)
    try:
        out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        handle.remove()
    return out.hidden_states, out.logits, captured_fit["fit"]


def subsequent_layers_mse(hidden_pred, hidden_target, layer_idx: int, n_resp: int, n_layers: int) -> torch.Tensor:
    """Sum of MSE at layer_idx and every layer after it, on the response-token span only. hidden_states[i]
    is the output of layer (i-1), so "layer_idx and all subsequent layers" means indices
    layer_idx+1 .. n_layers inclusive."""
    total = torch.tensor(0.0, device=hidden_pred[0].device)
    for idx in range(layer_idx + 1, n_layers + 1):
        pred = hidden_pred[idx][0, -n_resp:, :].float()
        target = hidden_target[idx][0, -n_resp:, :].float()
        total = total + F.mse_loss(pred, target)
    return total


@torch.no_grad()
def collect_target_hidden_states(model, full_instr: torch.Tensor):
    return model(input_ids=full_instr, output_hidden_states=True).hidden_states


def make_inference_hook(gate: GateState, direction: torch.Tensor):
    """Inference-time hook for use with steering.hooks.steering_hook. Distinguishes prefill (full
    prompt, seq_len > 1 -- don't steer, it's all prompt) from generation steps (seq_len == 1 -- the
    single new token IS the response by construction, always steer) -- same is_generating logic
    Nokia's reference implementation uses, adapted to HF's KV-cache generation loop."""
    @torch.no_grad()
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            return hidden
        coeff, _ = coefficient(gate, hidden, mask=None)
        return hidden + (coeff * direction.to(hidden.dtype)).to(hidden.dtype)
    return hook_fn
