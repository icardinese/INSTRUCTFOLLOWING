"""Batches MULTIPLE, DIFFERENT steering configurations into one forward/generate pass, instead of
one call per configuration. See BATCHED_STEERING.md at the repo root for the full architectural
writeup -- this docstring covers only the invariant that makes it safe.

THE CORE INVARIANT THIS RELIES ON: a decoder-only transformer, run without cross-batch operations
(no batch norm, no cross-attention between rows), computes each row of a batch completely
independently of every other row. Row i's hidden state at any layer depends only on row i's own
tokens and attention mask -- never on row j's. This is already true of every model this project
targets (Llama/Mistral/Gemma2/Phi-3/Qwen2 -- all decoder-only, all normalize per-row via
RMSNorm/LayerNorm across the FEATURE dimension, never across the batch dimension). Consequence:
applying a DIFFERENT correction to different rows of the SAME batched forward pass produces
EXACTLY the same result, row for row, as running each row through its own separate forward pass
with its own correction. Batching heterogeneous configs is not an approximation of running them
separately -- it's mathematically identical to it, just done in one pass instead of N.

What this file adds is the missing piece: EVERY existing single-config hook function in this
project (make_inference_hook in gate.py, conceptor/matrix/logic.py, conceptor/selfproj/logic.py)
already broadcasts correctly across a batch dimension of any size -- coefficient()'s matmul and
make_inference_hook's `hidden.shape[1] > 1` check both operate per-row already, with no
batch-size assumption anywhere. What's been missing is a way to apply a DIFFERENT one of these
existing hook functions to different SLICES of one batch. That's make_routed_hook, below --
nothing about the existing gate/coefficient/direction math changes.
"""
from typing import Any, Callable

import torch


def make_routed_hook(row_groups: dict[Any, list[int]], hooks_by_group: dict[Any, Callable]) -> Callable:
    """Builds one hook function that applies a DIFFERENT existing single-config hook to each named
    group's rows of a larger batch, leaving every other row untouched.

    row_groups: group_id -> list of row indices (into the batch's dim 0) belonging to that group.
    hooks_by_group: group_id -> hook_fn(hidden_slice) -> corrected_hidden_slice -- an EXISTING,
        unmodified hook such as gate.make_inference_hook(gate_state, direction). A group_id present
        in row_groups but ABSENT from hooks_by_group is a deliberate way to mix unsteered baseline
        rows into the same batch: those rows pass through with no correction at all.

    Each group's hook_fn only ever sees ITS OWN rows (shape (len(rows), seq_len, d)), never the
    full batch -- from that hook's point of view, it's running exactly as it would standalone.
    That's the whole trick: no hook function anywhere needs to know batching is happening.
    """
    def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
        out = hidden.clone()
        for group_id, rows in row_groups.items():
            fn = hooks_by_group.get(group_id)
            if fn is None:
                continue  # no correction registered for this group at this layer -- leave as-is
            out[rows] = fn(hidden[rows])
        return out
    return hook_fn


def left_pad_batch(tokenizer, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-padding is required (not a style choice) for batched causal-LM generation: the newest,
    still-growing token position must be the RIGHTMOST column for every row, or a correctly-decoded
    row and a still-padded row can't share the same "generate the next token" step. Restores
    whatever padding_side the tokenizer had before returning, since generation-time padding is a
    local concern of this call, not a global tokenizer setting other code should inherit."""
    original_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = original_side
    return enc["input_ids"], enc["attention_mask"]


def generate_batched_uniform(
    model,
    tokenizer,
    prompts: list[str],
    hooks: dict[int, Callable] | None = None,
    max_new_tokens: int = 150,
    max_batch_rows: int = 60,
) -> list[str]:
    """Batched generation where EVERY row gets the SAME intervention (or none). Complements
    generate_with_routed_configs, which exists for the opposite case -- different configs on
    different rows of one batch, one layer per group.

    This is what multi-layer methods (A-PSR / Multi-Gate) need: those hook ALL layers for every
    row, so `layer_by_group`'s one-layer-per-group model doesn't apply, and no routing is needed
    at all since there's nothing to distinguish rows by. Passing hooks=None generates unsteered
    (Base/Prompt baselines), so a caller can use one code path for every condition.

    Chunked at max_batch_rows for the same reason evals/layer_hparam_search.py's
    _chunk_groups_by_row_budget exists: an unbounded batch hit a real CUDA OOM on an 80GB A100.

    Returns responses in the SAME order as `prompts`.
    """
    from steering.hooks import multi_steering_hook  # local import: avoids a hooks<->batch_routing cycle

    responses: list[str] = []
    for start in range(0, len(prompts), max_batch_rows):
        chunk = prompts[start: start + max_batch_rows]
        input_ids, attention_mask = left_pad_batch(tokenizer, chunk)
        input_ids, attention_mask = input_ids.to(model.device), attention_mask.to(model.device)

        def _do_generate():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        if hooks:
            with multi_steering_hook(model, hooks):
                output_ids = _do_generate()
        else:
            output_ids = _do_generate()

        prompt_len = input_ids.shape[1]
        responses.extend(
            tokenizer.decode(output_ids[r, prompt_len:], skip_special_tokens=True).strip()
            for r in range(len(chunk))
        )
    return responses


def generate_with_routed_configs(
    model,
    tokenizer,
    prompts_by_group: dict[Any, list[str]],
    hook_fns_by_group: dict[Any, Callable],
    layer_by_group: dict[Any, int] | int,
    max_new_tokens: int = 150,
) -> dict[Any, list[str]]:
    """One generate() call serving every group at once. group_id is anything hashable -- an alpha
    value, a (layer, alpha, loss_config) tuple, "baseline" for unsteered rows, whatever the
    caller's grid actually is. A group omitted from hook_fns_by_group gets no correction (useful
    for mixing in Base/Prompt baseline rows alongside steered ones in the exact same batch).

    layer_by_group: EITHER a single int (every group hooks the same layer -- the common case for a
    hyperparameter sweep at one fixed layer, e.g. an alpha x loss-config grid) OR a
    {group_id: layer_idx} dict when different groups hook DIFFERENT layers (the layer-sweep case --
    e.g. comparing 13 layers at once). Both share one generate() call either way: a row belonging
    to a group hooked at layer 7 simply isn't present in layer 14's routed hook's row_groups dict
    at all, so layer 14's hook has no way to touch it even by accident.

    Returns {group_id: [response_per_prompt_in_that_group]}, same order as prompts_by_group.
    """
    group_ids = list(prompts_by_group.keys())
    flat_prompts: list[str] = []
    row_groups: dict[Any, list[int]] = {}
    for g in group_ids:
        start = len(flat_prompts)
        flat_prompts.extend(prompts_by_group[g])
        row_groups[g] = list(range(start, len(flat_prompts)))

    input_ids, attention_mask = left_pad_batch(tokenizer, flat_prompts)
    input_ids, attention_mask = input_ids.to(model.device), attention_mask.to(model.device)

    from steering.hooks import multi_steering_hook  # local import: avoids a hooks<->batch_routing cycle

    if isinstance(layer_by_group, int):
        layer_by_group = {g: layer_by_group for g in group_ids}

    # Invert group->layer into layer->groups, then build ONE routed hook per DISTINCT layer,
    # scoped to only the groups actually assigned to it.
    groups_by_layer: dict[int, list[Any]] = {}
    for g, layer_idx in layer_by_group.items():
        groups_by_layer.setdefault(layer_idx, []).append(g)

    hooks_by_layer = {}
    for layer_idx, groups_here in groups_by_layer.items():
        row_groups_here = {g: row_groups[g] for g in groups_here}
        hooks_here = {g: hook_fns_by_group[g] for g in groups_here if g in hook_fns_by_group}
        hooks_by_layer[layer_idx] = make_routed_hook(row_groups_here, hooks_here)

    with multi_steering_hook(model, hooks_by_layer):
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    responses_by_group: dict[Any, list[str]] = {}
    for g in group_ids:
        responses_by_group[g] = [
            tokenizer.decode(output_ids[r, prompt_len:], skip_special_tokens=True).strip()
            for r in row_groups[g]
        ]
    return responses_by_group
