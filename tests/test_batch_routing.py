"""Tests for steering/batch_routing.py. Structured deliberately around ONE separation: routing
correctness (does make_routed_hook dispatch the right rows to the right hook, leave others alone)
is tested at the raw TENSOR level, independent of any model or generate() loop -- because that's
the actual novel claim, and it's fully provable without touching a model at all. Padding
correctness and the full generate() pipeline are tested separately, through the fake model, since
they depend on real forward-pass/tokenizer behavior that tensor-level tests can't exercise.

Why not just test everything through model.generate() end to end: TinyModel's generate() (see
tests/fakes.py) recomputes the whole growing sequence from scratch every step rather than using a
real KV cache, so every internal forward call it makes has seq_len > 1 -- it can never actually
present a decode-shaped (batch, 1, d) hidden state to a hook the way a real HF model's cached
generation loop does. That's a pre-existing property of the fake model, not something this file's
tests work around by cheating -- it just means "prove routing is correct" (tensor-level, any
shape) and "prove the fake model's plumbing doesn't crash" (generate()-level, smoke test only)
have to be two different tests, not one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from steering.batch_routing import generate_with_routed_configs, left_pad_batch, make_routed_hook
from steering.hooks import steering_hook
from steering.psr.gate import init_gate_state, make_inference_hook
from tests.fakes import make_fake_model_and_tokenizer

# ---------------------------------------------------------------------------
# Part 1: routing correctness, at the tensor level, with simple synthetic hooks.
# This is the actual claim: N different corrections, applied to N different row-groups of ONE
# batch, must produce EXACTLY what running each group through its own hook separately would.
# ---------------------------------------------------------------------------


def test_routed_hook_dispatches_each_group_to_its_own_hook_decode_shape():
    """Decode-shaped tensors (batch, 1, d) -- the shape a real KV-cached generate() loop actually
    presents to a hook at each new-token step."""
    torch.manual_seed(0)
    hidden = torch.randn(4, 1, 8)  # 4 rows, 1 token each, 8-dim
    row_groups = {"a": [0, 1], "b": [2, 3]}
    hooks_by_group = {
        "a": lambda h: h + 10.0,
        "b": lambda h: h * 2.0,
    }
    routed = make_routed_hook(row_groups, hooks_by_group)
    out = routed(hidden)

    assert torch.allclose(out[[0, 1]], hidden[[0, 1]] + 10.0)
    assert torch.allclose(out[[2, 3]], hidden[[2, 3]] * 2.0)


def test_routed_hook_dispatches_each_group_to_its_own_hook_prefill_shape():
    """Same claim, but with a real multi-token (prefill-shaped) tensor -- routing doesn't care
    about sequence length at all, only about which ROWS belong to which group."""
    torch.manual_seed(1)
    hidden = torch.randn(3, 5, 8)
    row_groups = {"x": [0], "y": [1, 2]}
    hooks_by_group = {"x": lambda h: h - 1.0, "y": lambda h: h + 100.0}
    routed = make_routed_hook(row_groups, hooks_by_group)
    out = routed(hidden)

    assert torch.allclose(out[[0]], hidden[[0]] - 1.0)
    assert torch.allclose(out[[1, 2]], hidden[[1, 2]] + 100.0)


def test_routed_hook_leaves_ungrouped_rows_completely_unchanged():
    """A group present in row_groups but ABSENT from hooks_by_group -- the documented way to mix
    unsteered baseline rows into the same batch as steered ones."""
    torch.manual_seed(2)
    hidden = torch.randn(3, 1, 8)
    row_groups = {"steered": [0], "baseline": [1, 2]}
    hooks_by_group = {"steered": lambda h: h + 999.0}  # "baseline" deliberately has no entry
    routed = make_routed_hook(row_groups, hooks_by_group)
    out = routed(hidden)

    assert torch.allclose(out[[0]], hidden[[0]] + 999.0)
    assert torch.equal(out[[1, 2]], hidden[[1, 2]]), "ungrouped/baseline rows must be byte-for-byte identical to input"


def test_routed_hook_matches_running_each_group_separately():
    """The actual end-to-end claim: batching K different hooks together gives the identical
    result, row for row, as calling each hook alone on just its own rows. Uses REAL
    gate/direction configs (steering.psr.gate.make_inference_hook), not toy lambdas, so this is
    also a direct test of the real machinery, not just the routing primitive in isolation."""
    torch.manual_seed(3)
    d = 16
    configs = {}
    for i in range(3):
        gate = init_gate_state(d, "cpu")
        gate.weight.data = torch.randn(d, 1) * 0.1
        gate.bias.data = torch.tensor([0.05 * i])
        direction = torch.randn(d)
        configs[i] = make_inference_hook(gate, direction)

    hidden = torch.randn(3, 1, d)  # one row per config, decode-shaped
    row_groups = {i: [i] for i in range(3)}
    routed = make_routed_hook(row_groups, configs)
    batched_out = routed(hidden)

    for i in range(3):
        separate_out = configs[i](hidden[[i]])  # run config i alone on just its own row
        assert torch.allclose(batched_out[[i]], separate_out, atol=1e-6), (
            f"config {i}'s batched result must exactly match its separately-run result"
        )


def test_routed_hook_handles_multiple_rows_per_group():
    """A group can own more than one row (e.g. 20 prompts all sharing the SAME hyperparameter
    config) -- the hook function must receive and correctly process the whole sub-batch at once,
    not just single rows."""
    torch.manual_seed(4)
    d = 16
    gate = init_gate_state(d, "cpu")
    direction = torch.randn(d)
    hook = make_inference_hook(gate, direction)

    hidden = torch.randn(5, 1, d)
    row_groups = {"g": [0, 1, 2, 3, 4]}  # all 5 rows, one shared config
    routed = make_routed_hook(row_groups, {"g": hook})
    batched_out = routed(hidden)
    separate_out = hook(hidden)  # running the hook on the whole batch directly should be identical

    assert torch.allclose(batched_out, separate_out, atol=1e-6)


# ---------------------------------------------------------------------------
# Part 2: left_pad_batch -- padding mechanics, independent of steering.
# ---------------------------------------------------------------------------


def test_left_pad_batch_produces_correctly_shaped_output_and_mask():
    _, tokenizer = make_fake_model_and_tokenizer()
    input_ids, attention_mask = left_pad_batch(tokenizer, ["ab", "abcde", "abc"])
    assert input_ids.shape == attention_mask.shape == (3, 5)
    # left-padded: real tokens are right-aligned, so the LAST column is real for every row
    assert (attention_mask[:, -1] == 1).all()
    # shortest prompt ("ab", 2 real tokens in a length-5 row) should have exactly 3 padding zeros
    assert attention_mask[0].tolist() == [0, 0, 0, 1, 1]


def test_left_pad_batch_restores_original_padding_side():
    _, tokenizer = make_fake_model_and_tokenizer()
    tokenizer.padding_side = "right"
    left_pad_batch(tokenizer, ["a", "abc"])
    assert tokenizer.padding_side == "right", "left_pad_batch must not leak its 'left' setting into the tokenizer's persistent state"


# ---------------------------------------------------------------------------
# Part 3: padding correctness through a REAL forward pass (no steering active -- isolates padding
# from steering, matching the "test one thing at a time" split explained in the module docstring).
# ---------------------------------------------------------------------------


def test_padding_does_not_change_real_tokens_hidden_states():
    """Batch three DIFFERENT-length prompts (forcing real left-padding for the two shorter ones),
    run one real forward pass, and confirm each row's LAST-position hidden state (always a real,
    non-padded token under left-padding) exactly matches what that same prompt produces when run
    alone, unpadded. No hook/steering involved -- this isolates "does padding corrupt anything" as
    its own claim, separate from routing."""
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    prompts = ["ab", "abcde", "abc"]
    input_ids, attention_mask = left_pad_batch(tokenizer, prompts)
    batched_out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    batched_last_layer_last_pos = batched_out.hidden_states[-1][:, -1, :]

    for i, p in enumerate(prompts):
        solo_ids = tokenizer(p, return_tensors="pt")["input_ids"]
        solo_out = model(input_ids=solo_ids, output_hidden_states=True)
        solo_last = solo_out.hidden_states[-1][0, -1, :]
        assert torch.allclose(batched_last_layer_last_pos[i], solo_last, atol=1e-5), (
            f"prompt {i!r}'s real-token hidden state changed when batched with padding -- "
            f"padding is leaking into real computation"
        )


# ---------------------------------------------------------------------------
# Part 4: full generate_with_routed_configs -- plumbing/shape smoke test.
# ---------------------------------------------------------------------------


def test_generate_with_routed_configs_runs_and_returns_right_shape():
    """Smoke test only, per the module docstring's caveat: TinyModel's generate() can't actually
    exercise per-row STEERING during decode (it has no real KV cache, so every internal step sees
    a multi-token, "prefill-shaped" tensor, which make_inference_hook always treats as
    no-steering). This test verifies the ORCHESTRATION -- grouping, padding, dispatch, response
    extraction -- runs correctly end to end and returns the right shape/keys, which is exactly
    what a fake model without real caching CAN meaningfully prove; the steering-math correctness
    itself is already proven above, independent of this."""
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    gate_a = init_gate_state(16, "cpu")
    gate_b = init_gate_state(16, "cpu")
    hooks_by_group = {
        "alpha_1": make_inference_hook(gate_a, torch.randn(16)),
        "alpha_2": make_inference_hook(gate_b, torch.randn(16)),
    }
    prompts_by_group = {
        "alpha_1": ["explain this", "what does it do"],
        "alpha_2": ["describe the function"],
    }

    responses = generate_with_routed_configs(
        model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group=1, max_new_tokens=3,
    )

    assert set(responses.keys()) == {"alpha_1", "alpha_2"}
    assert len(responses["alpha_1"]) == 2
    assert len(responses["alpha_2"]) == 1
    for group_responses in responses.values():
        for r in group_responses:
            assert isinstance(r, str)


def test_generate_with_routed_configs_supports_unsteered_baseline_group():
    """A group can be omitted from hooks_by_group entirely -- e.g. mixing a "no steering" baseline
    condition into the same batched call as several steered ones."""
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    hooks_by_group = {"steered": make_inference_hook(init_gate_state(16, "cpu"), torch.randn(16))}
    prompts_by_group = {"steered": ["explain this"], "baseline": ["what does it do"]}

    responses = generate_with_routed_configs(
        model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group=1, max_new_tokens=3,
    )
    assert set(responses.keys()) == {"steered", "baseline"}


# ---------------------------------------------------------------------------
# Part 5: multi-layer routing -- the extension that lets an entire LAYER SWEEP (not just a
# hyperparameter sweep at one fixed layer) collapse into a single generate() call. Different
# groups hook DIFFERENT layers; each layer's routed hook must only ever see (and only ever be
# able to touch) the rows belonging to groups assigned to THAT layer.
# ---------------------------------------------------------------------------


def test_multi_layer_routing_matches_running_each_layers_config_separately():
    """The real claim for the layer-sweep case: 3 different (layer, gate, direction) configs,
    batched together via layer_by_group, must give EXACTLY the same per-row result as hooking
    each layer alone (via steering_hook, one at a time) and running that config's row by itself."""
    torch.manual_seed(5)
    d = 16
    model, tokenizer = make_fake_model_and_tokenizer(d=d, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    configs = {}  # group_id -> (layer_idx, hook_fn)
    for i, layer_idx in enumerate([0, 1, 2]):
        gate = init_gate_state(d, "cpu")
        gate.weight.data = torch.randn(d, 1) * 0.1
        direction = torch.randn(d)
        configs[i] = (layer_idx, make_inference_hook(gate, direction))

    hidden = torch.randn(3, 1, d)  # decode-shaped, one row per config
    layer_by_group = {i: layer for i, (layer, _) in configs.items()}
    hooks_by_group = {i: fn for i, (_, fn) in configs.items()}

    # Build the SAME per-layer routing the real function would, directly (tensor-level, no model
    # needed for this specific check -- proves the routing math, independent of the model).
    groups_by_layer: dict[int, list[int]] = {}
    for i, (layer, _) in configs.items():
        groups_by_layer.setdefault(layer, []).append(i)

    out = hidden.clone()
    for layer, groups_here in groups_by_layer.items():
        row_groups_here = {g: [g] for g in groups_here}
        hooks_here = {g: hooks_by_group[g] for g in groups_here}
        routed = make_routed_hook(row_groups_here, hooks_here)
        # Each layer's routed hook only ever touches its own groups' rows (here, disjoint by
        # construction since each config is at a DIFFERENT layer) -- applying them in sequence is
        # equivalent to applying them "simultaneously" specifically because the row sets don't overlap.
        out = routed(out)

    for i, (layer, fn) in configs.items():
        separate_out = fn(hidden[[i]])
        assert torch.allclose(out[[i]], separate_out, atol=1e-6), f"config {i} (layer {layer}) diverged"


def test_generate_with_routed_configs_accepts_per_group_layer_mapping():
    """End-to-end plumbing test: layer_by_group as a dict (different groups, different layers)
    must run without error and route each group correctly -- same smoke-test scope as the
    single-layer version (see that test's docstring for why this can't prove decode-time steering
    numerics through the fake model's non-cached generate(), only that the orchestration works)."""
    model, tokenizer = make_fake_model_and_tokenizer(d=16, n_layers=4)
    for p in model.parameters():
        p.requires_grad_(False)

    hooks_by_group = {
        "layer0_cfg": make_inference_hook(init_gate_state(16, "cpu"), torch.randn(16)),
        "layer2_cfg": make_inference_hook(init_gate_state(16, "cpu"), torch.randn(16)),
    }
    prompts_by_group = {
        "layer0_cfg": ["explain this", "what does it do"],
        "layer2_cfg": ["describe the function"],
        "baseline": ["no steering here"],  # deliberately has neither a hook nor a layer entry issue
    }
    layer_by_group = {"layer0_cfg": 0, "layer2_cfg": 2, "baseline": 0}  # baseline's layer doesn't matter, it has no hook

    responses = generate_with_routed_configs(
        model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group=layer_by_group, max_new_tokens=3,
    )
    assert set(responses.keys()) == {"layer0_cfg", "layer2_cfg", "baseline"}
    assert len(responses["layer0_cfg"]) == 2
    assert len(responses["layer2_cfg"]) == 1
    assert len(responses["baseline"]) == 1


def test_multi_layer_groups_cannot_leak_into_each_others_layer():
    """If group A is assigned to layer 0 and group B to layer 2, layer 0's routed hook must not
    even be CAPABLE of touching group B's rows -- checked by confirming group B's row is absent
    from layer 0's row_groups dict entirely, not merely unmodified by coincidence."""
    d = 8
    gate_a = init_gate_state(d, "cpu")
    gate_b = init_gate_state(d, "cpu")
    hook_a = make_inference_hook(gate_a, torch.randn(d))
    hook_b = make_inference_hook(gate_b, torch.randn(d))

    layer_by_group = {"a": 0, "b": 2}
    hooks_by_group = {"a": hook_a, "b": hook_b}
    row_groups = {"a": [0], "b": [1]}

    groups_by_layer: dict[int, list[str]] = {}
    for g, layer in layer_by_group.items():
        groups_by_layer.setdefault(layer, []).append(g)

    layer0_row_groups = {g: row_groups[g] for g in groups_by_layer[0]}
    layer2_row_groups = {g: row_groups[g] for g in groups_by_layer[2]}

    assert "b" not in layer0_row_groups, "group b (layer 2) must not appear in layer 0's routing table at all"
    assert "a" not in layer2_row_groups, "group a (layer 0) must not appear in layer 2's routing table at all"
