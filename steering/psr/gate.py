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
from steering.hooks import rewrap_hidden, unwrap_hidden
from steering.psr.reference_config import USE_COEFF_BIAS
from steering.psr.spans import ANSWER_ONLY, prefill_tail_length, steering_mask


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
        """Only tensors that actually require grad. coeff_bias is frozen at 0 by default (see
        init_gate_state), and handing a requires_grad=False leaf to an optimizer is at best a
        no-op and at worst an error depending on torch version -- so filter here rather than
        making all seven call sites remember to."""
        return [p for p in (self.weight, self.bias, self.coeff_bias) if p.requires_grad]


def init_gate_state(
    hidden_size: int,
    device: str,
    dtype: torch.dtype = torch.float32,
    use_coeff_bias: bool = USE_COEFF_BIAS,
) -> GateState:
    """coeff_bias defaults to FROZEN AT ZERO. The parameter is faithful -- it is b_{m,l} in the
    paper (Section 3.6) and the reference's FocusedSteeringModule computes
    `user_steering_coeffs + steering_coeff_bias`, which is exactly this project's
    `1.0 + coeff_bias` at alpha=1. But FocusedSteeredModelConfig's dataclass default of
    use_steering_coeff_bias=True is never used in any reported experiment:
    base_architecture_focused() sets it False, and experiments/llm_steer_instruct/eval.py sets
    it False again. Training it is therefore an extra degree of freedom relative to every
    published PSR number.

    Kept as a flag rather than deleted so the ablation ("does a learned global scale help?") is
    still one argument away, and so checkpoint format is unchanged -- coeff_bias is still saved,
    it is simply always 0.0."""
    weight = (torch.randn(hidden_size, 1, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
    coeff_bias = torch.zeros(1, device=device, dtype=dtype).requires_grad_(use_coeff_bias)
    return GateState(weight=weight, bias=bias, coeff_bias=coeff_bias)


def location_fit(gate: GateState, hidden: torch.Tensor) -> torch.Tensor:
    """How much THIS position needs steering. (batch, seq, d) -> (batch, seq, 1)."""
    return torch.relu(hidden.float() @ gate.weight + gate.bias)


def answer_only_mask(seq_len: int, n_resp: int, device) -> torch.Tensor:
    """R (response-only) mask: the final PROMPT token plus the response span, n_resp + 1 positions.

    The +1 is the reference's definition, not an off-by-one bug. Their mask is
    `token_positions >= last_input_token_positions` (constant_steering.py::compute_steering_mask),
    and last_input_token_position is the index of the last prompt token, not the first response
    token (tokenization_utils.compute_last_input_token_index returns len(prompt_tokens) - 1), so
    `>=` includes it. That position is the one whose hidden state produces the first generated
    token; excluding it (as this file did before 2026-09-20) meant the intervention never touched
    the decision that sets the tone for the whole response.

    Kept as a named wrapper because R is the default surface for caveman and IFEval and most call
    sites want it by name. For QR, or to make the surface configurable, call
    steering.psr.spans.steering_mask directly.

    response_nll, subsequent_layers_mse and make_inference_hook must all agree on this span."""
    return steering_mask(seq_len, n_resp, device, location=ANSWER_ONLY)


def coefficient(gate: GateState, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (coeff, fit). coeff is what every variant's correction gets scaled by; fit is exposed
    separately since regularization_loss needs it directly, not the scaled coefficient."""
    fit = location_fit(gate, hidden)
    if mask is not None:
        fit = torch.where(mask, fit, torch.zeros_like(fit))
    desired_presence = 1.0 + gate.coeff_bias
    return desired_presence * fit, fit


def regularization_loss(fit_vals: torch.Tensor | list[torch.Tensor], reg_coeff: float) -> torch.Tensor:
    """Penalizes the gate for firing NOWHERE across a whole response -- the dead-ReLU guard. Without
    this, relu(1 - sum(fit)) < 0 gives zero gradient pressure away from an all-zero (do-nothing) gate.

    Accepts either a single fit tensor (single-layer PSR) or a LIST of them (A-PSR / Multi-Gate,
    one per intervention layer), in which case the per-layer penalties are SUMMED -- matching
    Nokia's reference loop, which adds each intervention module's own regularization_loss into the
    total (`for layer_output in outputs.intervention_outputs.values(): loss += ...`). Each layer's
    gate needs its own independent dead-ReLU pressure; averaging instead of summing would let one
    live gate mask N-1 dead ones, and concatenating the fits into one tensor instead would pool
    all layers' positions into a single sum, which is a different (weaker) constraint entirely."""
    if isinstance(fit_vals, (list, tuple)):
        return sum(regularization_loss(f, reg_coeff) for f in fit_vals)
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
    alone.

    Uses steering.hooks.unwrap_hidden/rewrap_hidden rather than assuming `output` is always a
    tuple -- some transformers versions return the decoder layer's hidden_states as a plain
    tensor instead, and assuming a tuple there causes a real, confirmed crash (`AttributeError:
    'tuple' object has no attribute 'dtype'` several layers downstream, once the mis-wrapped
    tuple gets fed into the next layer as if it were the hidden-states tensor)."""
    layer = get_decoder_layers(model)[layer_idx]
    captured_fit = {}

    def wrapped(module, inputs, output):
        hidden, rest = unwrap_hidden(output)
        mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
        coeff, fit = coefficient(gate, hidden, mask)
        correction = (coeff * direction.to(hidden.dtype)).to(hidden.dtype)
        captured_fit["fit"] = fit
        return rewrap_hidden(hidden + correction, rest)

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
        pred = hidden_pred[idx][0, -(n_resp + 1):, :].float()
        target = hidden_target[idx][0, -(n_resp + 1):, :].float()
        total = total + F.mse_loss(pred, target)
    return total


@torch.no_grad()
def collect_target_hidden_states(model, full_instr: torch.Tensor):
    return model(input_ids=full_instr, output_hidden_states=True).hidden_states


def make_inference_hook(gate: GateState, direction: torch.Tensor, prefill_tail: int = 1):
    """Inference-time hook for use with steering.hooks.steering_hook.

    PREFILL IS NOT SKIPPED. The reference does not skip it either -- this was a misreading of
    their `is_generating` flag that stood in this file until 2026-09-20. Their logic is:

        is_generating = max_seq_len == 1
        steering_mask = compute_steering_mask(is_generating, ..., self.config.steering_location)

    When is_generating is False (prefill), they do not bail out; they apply the positional mask,
    and under steering_location="answer_only" that mask is `pos >= last_input_token_position`,
    which still selects the FINAL PROMPT TOKEN. So the reference steers
    {last prompt token} u {every generated token}. Skipping prefill entirely dropped the first
    of those, and left training (which does steer it, via answer_only_mask) disagreeing with
    inference.

    prefill_tail is how many TRAILING prefill positions to steer, and is what makes the R/QR
    surface switchable at inference. 1 == R (the final prompt token). For QR, pass
    steering.psr.spans.prefill_tail_length(prompt_len, "question_and_answer", last_sys_idx),
    which counts from the right so it stays correct for every row of a left-padded batch.

    Counting from the right is safe precisely because steering/batch_routing.py::left_pad_batch
    left-pads, right-aligning every row. Under right padding these positions would be pad tokens
    for the shorter rows -- do not change the padding side without revisiting this."""
    @torch.no_grad()
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            # Prefill: steer the trailing `prefill_tail` positions (1 for R, the whole
            # question span for QR).
            k = min(prefill_tail, hidden.shape[1])
            tail = hidden[:, -k:, :]
            coeff, _ = coefficient(gate, tail, mask=None)
            out = hidden.clone()
            out[:, -k:, :] = tail + (coeff * direction.to(hidden.dtype)).to(hidden.dtype)
            return out
        # Decode: the single new token IS a response token by construction.
        coeff, _ = coefficient(gate, hidden, mask=None)
        return hidden + (coeff * direction.to(hidden.dtype)).to(hidden.dtype)
    return hook_fn
