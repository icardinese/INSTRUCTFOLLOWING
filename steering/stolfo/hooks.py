"""Stolfo et al. 2025's ACTUAL proposed mechanism (direction_projection_hook / adjust_vectors in
microsoft/llm-steer-instruct's utils/generation_utils.py), verified against that repo directly --
NOT the same as this project's Const condition, which implements their OTHER, simpler baseline
(activation_addition_hook). Confirmed via format/find_best_layer.py:

    instr_dir = last_token_mean_diff[layer_idx] / last_token_mean_diff[layer_idx].norm()
    proj = hs_instr[:, layer_idx, -1, :] @ instr_dir
    avg_proj = proj.mean()
    hook_fn = functools.partial(direction_projection_hook, direction=instr_dir, value_along_direction=avg_proj)

And direction_projection_hook itself:
    current_projections = v @ u
    delta = target_values - current_projections
    adjusted_v = v + delta[:, None] * u

So: h' = h + (target - h.u)*u -- the component of h along u gets REPLACED by target, not offset by
a fixed amount. The correction magnitude is a genuine (closed-form, non-learned) function of the
CURRENT activation's own projection -- distinct from both Const (fixed offset regardless of state)
and this project's PSR gate (a TRAINED probe mapping activation -> coefficient).

Applied UNCONDITIONALLY at every position, matching their reference generate_with_hooks exactly:
that function has no KV-cache and reprocesses the whole sequence-so-far on every generation step,
so the hook fires on every position (prompt AND every generated token) every single call -- not
"last token only" at application time, even though extraction (the direction/target themselves)
IS last-token-only. Do NOT add an is_generating/seq_len==1 gate here the way steering/psr/gate.py
does -- that would silently change the method being replicated.

One real, documented divergence from their literal reference loop: this project's generation path
uses HF's KV-cached model.generate() (steering.hooks.steering_hook fires on whatever `hidden` shape
a given forward call produces, unmodified), so in practice the clamp fires once over the whole
prompt during prefill, then once per NEW token during decode -- it does NOT re-clamp
already-generated earlier tokens on every subsequent step the way their uncached from-scratch loop
literally does (each of their steps reprocesses the ENTIRE sequence so far). Re-clamping stale
positions repeatedly has no discernible reason to change the OUTPUT (their delta only ever depends
on that position's own current activation and the fixed target -- clamping the same fixed prompt
tokens to the same target repeatedly is idempotent, not cumulative), but it is a real speed
tradeoff being made deliberately here, not an oversight -- flagging it so it doesn't get "verified"
as identical without this caveat.
"""
from typing import Callable

import torch


def make_stolfo_projection_hook(direction: torch.Tensor, target_projection: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """direction: unit vector (d,). target_projection: scalar (Python float or 0-d tensor) -- the
    instructed condition's mean projection onto direction, computed once, not per-example."""
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        d = direction.to(hidden.dtype).to(hidden.device)
        current_proj = hidden.float() @ d.float()  # [batch, seq_len]
        delta = (float(target_projection) - current_proj).to(hidden.dtype)  # [batch, seq_len]
        return hidden + delta.unsqueeze(-1) * d

    return hook_fn
