"""Projection-clamp steering, with intervention SURFACE and GATING as explicit, independent knobs.

    h' = h + w(h) * (tau - h.u) * u

where u is a unit direction, tau a fixed target projection, and w(h) the gate weight (identically
1 when ungated). The bracketed term is what makes this a clamp rather than a translation: it
measures each activation's CURRENT projection onto u and moves exactly the shortfall, so after an
ungated application every position satisfies h'.u == tau exactly, regardless of where it started.
An additive intervention (h + c*u) cannot do that -- a fixed c has to be tuned to an average
activation and will overshoot some and undershoot others.

WHY THE SURFACE FLAG EXISTS. Stolfo et al.'s reference implementation clamps EVERY position,
prompt included (their generate_with_hooks has no KV-cache and no position mask). Every method in
the PSR family does the opposite: steering/psr/gate.py's make_inference_hook returns prefill
untouched (`if hidden.shape[1] > 1: return hidden`) and answer_only_mask zeroes the gate outside
the response span. That is a real confound in any Stolfo-vs-PSR comparison -- the two differ in
functional form AND in how much of the sequence they touch -- and it cannot be untangled from
results where both vary at once.

It matters because of the KV cache: modifying prompt positions during prefill changes the keys and
values cached for those positions at every subsequent layer, so all later generated tokens attend
back to a rewritten context. Response-only steering leaves the context intact and nudges only each
new token's own residual stream.

`response_only=True` reproduces PSR's surface exactly (same prefill guard), letting the form be
compared at matched surface. `response_only=False` reproduces Stolfo's. Note that with HF's
KV-cached generation the False case clamps the prompt once at prefill and each new token once at
decode, rather than re-clamping the whole growing sequence every step as their uncached loop does;
the clamp is idempotent under greedy decoding, so the outputs match, but the work differs.
"""
from typing import Callable

import torch

from steering.psr.gate import GateState, coefficient


def clamp_delta(hidden: torch.Tensor, direction: torch.Tensor, target: float) -> torch.Tensor:
    """(tau - h.u) for every position -> shape [batch, seq_len, 1], ready to scale `direction`.
    Computed in float32 regardless of the model's dtype: the shortfall is a difference of two
    similar-magnitude quantities, which is exactly where bf16 loses precision."""
    d = direction.float().to(hidden.device)
    proj = hidden.float() @ d
    return (float(target) - proj).unsqueeze(-1)


def make_clamp_hook(
    direction: torch.Tensor, target: float, response_only: bool = False
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Ungated clamp. response_only=False is Stolfo et al. as published."""
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        if response_only and hidden.shape[1] > 1:
            # Prefill (the whole prompt) -- identical guard to make_inference_hook's.
            return hidden
        d = direction.to(hidden.dtype).to(hidden.device)
        return hidden + (clamp_delta(hidden, direction, target).to(hidden.dtype) * d)

    return hook_fn


def make_gated_clamp_hook(
    gate: GateState, direction: torch.Tensor, target: float, response_only: bool = True
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Clamp scaled by PSR's learned gate: the probe decides WHERE to clamp, the closed-form
    shortfall decides HOW FAR. w(h) == 1 recovers the ungated clamp at that position, w(h) == 0
    leaves it untouched.

    Deliberately reuses steering.psr.gate.coefficient -- the same relu(w.h + b) * (1 + coeff_bias)
    the whole PSR family uses -- so "does PSR's gating help the clamp" is answered with PSR's
    actual gate rather than a bespoke one. Two consequences worth knowing:

      - relu is UNBOUNDED above, so w(h) > 1 can overshoot past the target (landing on the far
        side of the hyperplane). That is a real possibility, not a bug: the gate is free to learn
        that overshooting helps. If it turns out to be harmful, switching to a sigmoid bounds the
        weight to [0, 1] ("fraction of the way to the target") and is a one-line change here.
      - At initialization relu(w.h + b) is near zero, so training STARTS near "no intervention"
        and has to learn its way up to the clamp, rather than starting at Stolfo and learning
        where to back off. That's the same starting point every other gated variant in this
        project uses, which keeps the comparison fair.

    response_only defaults to True here (PSR's surface), since the point of gating is to compare
    against the gated PSR family; pass False to gate Stolfo's original surface instead.
    """
    @torch.no_grad()
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        if response_only and hidden.shape[1] > 1:
            return hidden
        coeff, _ = coefficient(gate, hidden, mask=None)
        d = direction.to(hidden.dtype).to(hidden.device)
        delta = clamp_delta(hidden, direction, target).to(hidden.dtype)
        return hidden + (coeff * delta * d)

    return hook_fn


def forward_with_gated_clamp_hook(
    model, gate: GateState, direction: torch.Tensor, target: float,
    layer_idx: int, input_ids: torch.Tensor, n_resp: int,
):
    """Training-time counterpart to make_gated_clamp_hook: one live, gradient-tracked forward pass
    with the gated clamp active at layer_idx. Returns (hidden_states, logits, fit) -- the same
    3-tuple steering/psr/training_loop.py's train_gate expects from any forward_fn, so the clamp
    trains through the identical loop (same epochs, masking, subsequent-layers MSE, dead-gate
    regularization) as every additive variant.

    Mirrors steering/psr/gate.py's forward_with_gate_hook exactly apart from the correction term:
    that one applies coeff * direction, this one applies coeff * (tau - h.u) * direction. Same
    unwrap/rewrap handling (decoder layers don't always return tuples) and the same
    answer_only_mask, so the gate only ever learns to fire on response positions.
    """
    from core.model_common import get_decoder_layers
    from steering.hooks import rewrap_hidden, unwrap_hidden
    from steering.psr.gate import answer_only_mask

    captured = {}
    layer = get_decoder_layers(model)[layer_idx]

    def wrapped(module, inputs, output):
        hidden, rest = unwrap_hidden(output)
        mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
        coeff, fit = coefficient(gate, hidden, mask)
        d = direction.to(hidden.dtype)
        delta = clamp_delta(hidden, direction, target).to(hidden.dtype)
        captured["fit"] = fit
        return rewrap_hidden(hidden + (coeff * delta * d), rest)

    handle = layer.register_forward_hook(wrapped)
    try:
        out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        handle.remove()
    return out.hidden_states, out.logits, captured["fit"]


def forward_with_multi_gated_clamp_hook(
    model, gates: dict, directions: dict, targets: dict,
    layer_indices: list, input_ids: torch.Tensor, n_resp: int,
):
    """MG+Clamp: gated clamps active at every layer in layer_indices simultaneously, one shared
    forward pass, one backward. Returns fit as a LIST (one per layer) -- regularization_loss sums
    over it, same as steering/psr/multi_gate.py. Each layer gets its OWN direction and target,
    since both are extracted per-layer.

    The closure factory is required for the same reason multi_gate.py documents: without it
    Python's late binding would make every hook use the last loop iteration's gate.
    """
    from core.model_common import get_decoder_layers
    from steering.hooks import rewrap_hidden, unwrap_hidden
    from steering.psr.gate import answer_only_mask

    decoder_layers = get_decoder_layers(model)
    captured: dict = {}
    handles = []

    for idx in layer_indices:
        def _make(gate, direction, target, i):
            def wrapped(module, inputs, output):
                hidden, rest = unwrap_hidden(output)
                mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
                coeff, fit = coefficient(gate, hidden, mask)
                d = direction.to(hidden.dtype)
                delta = clamp_delta(hidden, direction, target).to(hidden.dtype)
                captured[i] = fit
                return rewrap_hidden(hidden + (coeff * delta * d), rest)
            return wrapped
        handles.append(decoder_layers[idx].register_forward_hook(
            _make(gates[idx], directions[idx], targets[idx], idx)))

    try:
        out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    return out.hidden_states, out.logits, [captured[i] for i in layer_indices if i in captured]


def make_multi_gated_clamp_hooks(
    gates: dict, directions: dict, targets: dict, layer_indices: list, response_only: bool = True
) -> dict:
    """Inference-time {layer: hook_fn} dict for MG+Clamp, shaped for steering.hooks.multi_steering_hook."""
    return {
        idx: make_gated_clamp_hook(gates[idx], directions[idx], targets[idx], response_only=response_only)
        for idx in layer_indices
    }
