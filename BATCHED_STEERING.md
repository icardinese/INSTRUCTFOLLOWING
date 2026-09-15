# Batched heterogeneous steering: one forward pass, many different corrections

`steering/batch_routing.py` · tested in `tests/test_batch_routing.py`

## The problem this solves

Autoregressive generation is memory-bandwidth-bound, not compute-bound: at batch size 1, every
decode step reloads the *entire* model's weights from HBM to do one tiny matmul per token. An
A100's compute cores mostly sit idle waiting on memory, not computing. Batching N prompts into one
`generate()` call pays that same memory cost once and does N times the useful work in it — that's
why batching is the single biggest lever for generation throughput, full stop.

But a layer/hyperparameter sweep isn't N *copies of the same prompt* — it's N *different steering
configurations* (different gate weights, different directions, sometimes different layers)
applied to possibly-the-same or possibly-different prompts. The naive read of "just batch it" is
"batch same-config prompts together," which only helps within one config at a time — you still pay
the full sequential cost across configs. This document is about batching *across* configs too.

## The insight that makes it not just possible, but mathematically exact

A decoder-only transformer (every model this project targets — Llama, Mistral, Gemma2, Phi-3,
Qwen2) computes each row of a batch **completely independently of every other row**. There is no
cross-attention between batch elements, and every normalization (RMSNorm, LayerNorm) normalizes
across the *feature* dimension of a single row, never across the batch dimension. Row `i`'s hidden
state at any layer is a pure function of row `i`'s own tokens and mask — it has never, at any
point, been mixed with row `j`'s.

Consequence, stated precisely: **if you apply a different additive correction to different rows of
one batched forward pass, the result is not an approximation of running each row separately — it
is identical to it, bit for bit (up to floating-point associativity).** Batching heterogeneous
configs together doesn't trade correctness for speed. There's no tradeoff to make.

This is also not a novel claim invented for this project — it's the same principle production
multi-tenant LLM serving uses for batching many different LoRA adapters into one forward pass
(S-LoRA, Punica): different small interventions, one shared decode loop, per-row routing decides
which intervention applies where.

## What already existed vs. what was actually missing

Before this file, every single-config hook in this codebase — `make_inference_hook` in
`steering/psr/gate.py`, `conceptor/matrix/logic.py`, `conceptor/selfproj/logic.py` — *already*
worked correctly on any batch size:

```python
def hook_fn(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.shape[1] > 1:      # checks SEQUENCE length, not batch size
        return hidden
    coeff, _ = coefficient(gate, hidden, mask=None)   # broadcasts over any batch dim already
    return hidden + (coeff * direction).to(hidden.dtype)
```

`hidden.shape[1]` is the sequence-length dimension. Nothing here has ever assumed batch size 1 —
you could already call `make_inference_hook(gate, direction)` on a batch of 50 identical-config
rows and get correct results. **What was missing wasn't batch support in the math — it was a way
to point different rows of one batch at different (gate, direction) pairs.** That gap is exactly
`make_routed_hook`.

## The architecture

Three pieces, each doing exactly one job:

### 1. `make_routed_hook(row_groups, hooks_by_group)` — the actual novel part

```python
def make_routed_hook(row_groups, hooks_by_group):
    def hook_fn(hidden):
        out = hidden.clone()
        for group_id, rows in row_groups.items():
            fn = hooks_by_group.get(group_id)
            if fn is None:
                continue                      # no hook for this group -- leave those rows alone
            out[rows] = fn(hidden[rows])       # slice out this group's rows, run its OWN hook, write back
        return out
    return hook_fn
```

`hidden[rows]` (fancy indexing) pulls out exactly this group's rows as their own fresh sub-batch.
`fn` — an *existing, completely unmodified* `make_inference_hook(...)` closure — runs on that
sub-batch exactly as if it had been called alone; it has no idea it's part of something bigger.
`out[rows] = ...` writes the corrected sub-batch back into the right slots of the full batch.
Groups with no entry in `hooks_by_group` — a deliberate, documented feature — pass through
unchanged, which is how you mix an unsteered baseline condition into the same batched call as
several steered ones.

### 2. `left_pad_batch(tokenizer, prompts)` — padding, the unglamorous prerequisite

Different prompts have different lengths. Causal-LM batched generation requires **left**-padding
specifically (not right): the newest, still-decoding token position has to be the rightmost column
for every row simultaneously, or a fully-decoded row and a still-mid-decode row can't take the
"generate the next token" step together. This is mechanical, not clever — the only thing worth
knowing is *why* left, not right.

### 3. `generate_with_routed_configs(...)` — ties both together into one call

Flattens `{group_id: [prompts]}` into one padded batch, builds the row→group routing table,
registers `make_routed_hook(...)` at the target layer via the existing `steering.hooks.steering_hook`
context manager (unchanged — it already just calls `hook_fn(hidden)` and expects a same-shaped
tensor back, which is exactly what a routed hook is), calls `model.generate()` **once**, and
un-flattens the responses back into `{group_id: [responses]}`.

## Worked example: hyperparameter sweep at one fixed layer

Comparing 5 alpha values for `conceptor/matrix`, all hooked at layer 14, 20 prompts each:

```python
hooks_by_group = {
    alpha: make_inference_hook(gate, conceptor_direction_for(alpha))
    for alpha in [1.0, 2.0, 4.0, 8.0, 16.0]
}
prompts_by_group = {alpha: caveman_prompts[:20] for alpha in hooks_by_group}

responses = generate_with_routed_configs(
    model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group=14, max_new_tokens=150,
)
# responses[4.0] -> the 20 responses generated under alpha=4.0, from ONE generate() call
# that also produced responses[1.0], responses[2.0], responses[8.0], responses[16.0]
```

One decode loop, five configurations, at once. What would have been 5 sequential `generate()`
calls (5 × 20 = 100 sequential single-prompt generations, or 5 batched calls of 20) becomes one
batched call of 100 rows — the memory-bandwidth cost of loading the model gets paid once instead
of five times.

## Worked example: a full LAYER sweep, different layers, one call

This is the case that matters most for a rigorous layer sweep: `layer_by_group` can be a
`{group_id: layer_idx}` mapping instead of one shared int. Each distinct layer gets its OWN routed
hook (via `steering.hooks.multi_steering_hook`, already fixed earlier tonight for the plain-tensor
decoder-output bug), scoped to only the groups assigned to it:

```python
hooks_by_group = {layer: make_inference_hook(gate_for(layer), direction_for(layer)) for layer in candidate_layers}
prompts_by_group = {layer: caveman_prompts[:20] for layer in candidate_layers}
layer_by_group = {layer: layer for layer in candidate_layers}  # each group IS its own layer here

responses = generate_with_routed_configs(
    model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group=layer_by_group, max_new_tokens=150,
)
# one generate() call produced all 13 layers' worth of responses at once
```

A row assigned to layer 7 is never even present in layer 14's routing table — not filtered out at
call time, structurally absent, so there's no path by which one layer's correction could leak into
another layer's rows.

## How this was verified (and how you should read the test file)

The mathematical claim above — "batched with routing == running each config separately, row for
row" — is proven directly, at the tensor level, with **real** `make_inference_hook` configs
(`test_routed_hook_matches_running_each_group_separately`), not toy stand-ins. It's checked
deliberately in both the shape a real KV-cached decode step presents (`(batch, 1, d)`) and the
shape a prefill presents (`(batch, seq_len, d)`), since the router has to be correct in both.

Padding correctness is checked *separately*, through one real forward pass with genuinely
different-length prompts, confirming the last (always-real, under left-padding) position's hidden
state is identical whether that prompt was run alone or batched with padding
(`test_padding_does_not_change_real_tokens_hidden_states`).

These are split into different tests on purpose, not merged into one "does everything work" test —
see the module docstring in `test_batch_routing.py` for exactly why (short version: the fake
model's `generate()` has no real KV cache, so it can't itself exercise the decode-shape branch
through a full `generate()` call — the routing claim has to be proven independent of that
limitation, at the tensor level, rather than faked around).

Every test was also verified to actually *catch* a broken implementation before being trusted: the
routing logic was deliberately broken (made to apply one group's hook to the whole batch,
ignoring the routing table) and confirmed that the equivalence tests fail with a real, informative
assertion — then the fix was restored and the full suite re-confirmed green. Same discipline as
every other fix in this project's history: a test that can't fail isn't evidence of anything.

## Known limits — real, not hidden

- **Not yet run against the real model.** Every test above uses the fake model
  (`tests/fakes.py`) — appropriate for proving the routing *logic* is correct, since that claim
  depends only on tensor shapes and indexing, not on which model produced the tensors. It has not
  yet been run against the real Qwen2.5-Coder-7B-Instruct end to end. Before trusting this for a
  real sweep, run one small real comparison: batch 3-4 real configs together, compare against the
  same configs run separately (unbatched) on the real model, confirm the generated *text* matches
  token-for-token. The reasoning above says it must; that's still not the same as having watched
  it happen on the real model once. `scripts/verify_batch_routing_real_model.py` does exactly this.
- **Per-config batch size is still capped by GPU memory**, same as any batched generation — a
  13-layer sweep at n=20 each is 260 rows in one call; that's fine on an 80GB card for a 7B model,
  but it's a real ceiling, not an unlimited one. If a batch doesn't fit, split by however many
  groups fit at once rather than assuming this scales without bound.
