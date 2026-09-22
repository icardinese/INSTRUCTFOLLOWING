"""Multi-layer PSR mechanism -- the A-PSR ("all-layer") analogue of steering/psr/gate.py's
single-layer forward_with_gate_hook. Pure mechanism: this file has NO opinion on where the
directions come from (gradient-trained, diff-in-means, anything else) or which loss is used. That
keeps A-PSR and its Multi-Gate ablation on one shared code path, differing only in what the caller
hands in -- see src/psr/all_layer/train.py's --direction-source flag.

Verified against Nokia's reference implementation (Nokia-Bell-Labs/steer-like-the-llm). The key
finding, which is why this file is small: A-PSR is NOT a separate architecture in their code. It is
literally the same FocusedSteeredModelConfig with `layers=list(range(num_layers))` instead of
`layers=[layer]` (see their experiments/llm_steer_instruct/eval.py's "focused_all_layers" config
vs. the single-layer configs). Their SteeredModelBase.create_probe_interventions builds one
FocusedSteeringModule per layer, IntervenedModel registers them all as simultaneous hooks,
forward_pass runs ONE pass through all of them, train_steering_modules calls loss.backward() ONCE,
and optim.AdamW(intervened_model.parameters()) updates every layer's (gate, direction) jointly.
The "jointly optimizes simultaneous interventions in one shared forward pass" property therefore
falls out of hooking N layers in one pass and backpropagating once -- it needs no special
machinery, which is exactly what steering/hooks.py's multi_steering_hook docstring already said.

Two properties worth being explicit about, since both are load-bearing for faithfulness:

1. CHAINED, NOT INDEPENDENT. Each layer's gate sees the activation that ACTUALLY arrives there,
   already carrying every correction applied by earlier hooked layers, because all N hooks run
   inside one real forward pass. This is the "iteratively apply the intervention at all layers"
   behavior from the paper. It is the whole point, and it is what makes this genuinely different
   from the pre-existing mislabeled "A-PSR" in src/psr/old_baseline/train_a_psr.py, which trained
   N probes SEPARATELY on fixed precomputed activation pairs with zero interaction between layers.

2. ONE OPTIMIZER, ONE BACKWARD. This file doesn't own the optimizer, but it returns per-layer fit
   values in a single list precisely so the caller can build ONE loss over all layers and call
   backward once (steering/psr/gate.py's regularization_loss already accepts that list and sums
   per-layer penalties, matching Nokia's `for layer_output in ...: loss += regularization_loss`).
   Training each layer's gate in a separate backward pass would NOT be A-PSR.
"""
import torch

from core.model_common import get_decoder_layers
from steering.hooks import rewrap_hidden, unwrap_hidden
from steering.psr.gate import GateState, answer_only_mask, coefficient


def forward_with_multi_gate_hook(
    model,
    gates: dict[int, GateState],
    directions: dict[int, torch.Tensor],
    layer_indices: list[int],
    input_ids: torch.Tensor,
    n_resp: int,
):
    """Live, gradient-tracked forward pass with a gate correction active at EVERY layer in
    layer_indices simultaneously. Returns (hidden_states, logits, fit_vals) -- the same 3-tuple
    shape steering/psr/training_loop.py's train_gate already expects from any forward_fn, except
    fit_vals is a LIST (one tensor per hooked layer, ordered by layer_indices) rather than a single
    tensor. regularization_loss handles both shapes, so train_gate needs no changes at all.

    gates/directions are dicts keyed by layer index (not lists) so a caller can hook a subset of
    layers without silently relying on positional alignment.
    """
    decoder_layers = get_decoder_layers(model)
    captured_fit: dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layer_indices:
        def _make_wrapped(gate: GateState, direction: torch.Tensor, idx: int):
            # Factory function is required here for the same reason multi_steering_hook needs one:
            # without it every hook would close over the same loop variables by reference (Python's
            # late-binding closures), so all N layers would apply whichever gate/direction was
            # bound LAST -- silently training one layer's parameters N times instead of N layers'.
            def wrapped(module, inputs, output):
                hidden, rest = unwrap_hidden(output)
                mask = answer_only_mask(hidden.shape[1], n_resp, hidden.device)
                coeff, fit = coefficient(gate, hidden, mask)
                correction = (coeff * direction.to(hidden.dtype)).to(hidden.dtype)
                captured_fit[idx] = fit
                return rewrap_hidden(hidden + correction, rest)

            return wrapped

        handles.append(
            decoder_layers[layer_idx].register_forward_hook(
                _make_wrapped(gates[layer_idx], directions[layer_idx], layer_idx)
            )
        )

    try:
        out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    fit_vals = [captured_fit[idx] for idx in layer_indices if idx in captured_fit]
    return out.hidden_states, out.logits, fit_vals


def make_multi_inference_hooks(
    gates: dict[int, GateState],
    directions: dict[int, torch.Tensor],
    layer_indices: list[int],
    prefill_tail: int = 1,
) -> dict[int, callable]:
    """Inference-time counterpart, shaped for steering.hooks.multi_steering_hook (which takes a
    {layer_idx: hook_fn} dict). Mirrors steering/psr/gate.py's make_inference_hook exactly,
    including its prefill handling -- see that function's docstring for why prefill steers the
    final prompt token rather than being skipped. The two must stay identical: a divergence here
    would make A-PSR and S-PSR incomparable at inference for reasons unrelated to the
    architecture under test.

    No torch.no_grad() decorator here on purpose -- make_inference_hook has one, but it's applied
    at the hook level there; callers of this function are expected to already be inside
    torch.no_grad() (generation always is), and decorating here would silently break any future
    gradient-requiring use of the same hooks.
    """
    hooks = {}
    for layer_idx in layer_indices:
        def _make_hook(gate: GateState, direction: torch.Tensor):
            def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
                if hidden.shape[1] > 1:
                    # Prefill: steer the trailing `prefill_tail` positions (1 == R, question
                    # span == QR). Left padding right-aligns rows, so counting from the right
                    # is valid for every row of a batch.
                    k = min(prefill_tail, hidden.shape[1])
                    tail = hidden[:, -k:, :]
                    coeff, _ = coefficient(gate, tail, mask=None)
                    out = hidden.clone()
                    out[:, -k:, :] = tail + (coeff * direction.to(hidden.dtype)).to(hidden.dtype)
                    return out
                coeff, _ = coefficient(gate, hidden, mask=None)
                return hidden + (coeff * direction.to(hidden.dtype)).to(hidden.dtype)

            return hook_fn

        hooks[layer_idx] = _make_hook(gates[layer_idx], directions[layer_idx])
    return hooks
