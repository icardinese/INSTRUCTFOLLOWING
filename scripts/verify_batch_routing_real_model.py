"""Verify batched heterogeneous steering against the REAL model: run K different (layer, gate,
direction) configs BOTH batched-together (one generate_with_routed_configs call) AND separately
(one steering_hook + generate() call per config, exactly the pre-batching code path), then confirm
the generated text matches token-for-token, row for row.

This is the real-hardware confirmation BATCHED_STEERING.md's "Known limits" section calls for --
every test in tests/test_batch_routing.py proves the routing LOGIC is correct using the fake model
(appropriate, since that claim is pure tensor-shape/indexing math, independent of which model
produced the tensors), but "the reasoning says it must match" is not the same as having watched it
match on the actual model this project targets. Run this once before trusting batching for a real
sweep or judged-eval pass.

Usage (from repo root, on the GPU box):
    PYTHONPATH=.:src python3 scripts/verify_batch_routing_real_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.model_common import load_model, num_layers
from steering.batch_routing import generate_with_routed_configs
from steering.hooks import steering_hook
from steering.psr.gate import init_gate_state, make_inference_hook

# Small, deliberately real, deliberately varied -- different layers AND different prompt lengths,
# since both are exactly the things batching (multi-layer routing + left-padding) could get wrong
# if the reasoning in BATCHED_STEERING.md were mistaken.
PROMPTS = [
    "Explain what this function does:\ndef add(a, b):\n    return a + b",
    "Explain what this function does, in one sentence:\ndef is_even(n):\n    return n % 2 == 0",
    "What does this code do:\ndef square(x):\n    return x * x",
]
MAX_NEW_TOKENS = 40  # short on purpose -- this is a correctness check, not a real eval


def build_configs(hidden_size: int, n_layers: int, seed: int = 0):
    """A handful of real (layer, gate, direction) configs spanning different layers -- exactly the
    multi-layer routing case, not just same-layer hyperparameters."""
    torch.manual_seed(seed)
    layers = sorted({max(1, n_layers // 4), n_layers // 2, min(n_layers - 2, 3 * n_layers // 4)})
    configs = {}
    for i, layer_idx in enumerate(layers):
        gate = init_gate_state(hidden_size, "cuda")
        gate.weight.data = torch.randn(hidden_size, 1, device="cuda") * 0.05
        direction = torch.randn(hidden_size, device="cuda")
        configs[f"layer{layer_idx}"] = {
            "layer_idx": layer_idx,
            "hook_fn": make_inference_hook(gate, direction),
        }
    return configs


@torch.no_grad()
def generate_separately(model, tokenizer, configs: dict, prompts: list[str]) -> dict[str, list[str]]:
    """The pre-batching code path: one steering_hook + one generate() call PER prompt PER config --
    the actual baseline this script is checking batching against, not a hypothetical."""
    results = {}
    for group_id, cfg in configs.items():
        responses = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with steering_hook(model, cfg["layer_idx"], cfg["hook_fn"]):
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        results[group_id] = responses
    return results


@torch.no_grad()
def generate_batched(model, tokenizer, configs: dict, prompts: list[str]) -> dict[str, list[str]]:
    """The new code path: every config, every prompt, ONE generate() call."""
    prompts_by_group = {g: list(prompts) for g in configs}
    hooks_by_group = {g: cfg["hook_fn"] for g, cfg in configs.items()}
    layer_by_group = {g: cfg["layer_idx"] for g, cfg in configs.items()}
    return generate_with_routed_configs(
        model, tokenizer, prompts_by_group, hooks_by_group, layer_by_group, max_new_tokens=MAX_NEW_TOKENS,
    )


def main():
    print("Loading model...")
    model, tokenizer = load_model("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    hidden_size = model.config.hidden_size
    n_layers = num_layers(model)

    configs = build_configs(hidden_size, n_layers)
    layer_summary = ", ".join(f"{g}={c['layer_idx']}" for g, c in configs.items())
    print(f"Configs (layer): {layer_summary}, {len(PROMPTS)} prompts each, "
          f"{len(configs) * len(PROMPTS)} total generations per pass.\n")

    print("Running SEPARATELY (pre-batching code path, one generate() call per prompt per config)...")
    separate = generate_separately(model, tokenizer, configs, PROMPTS)

    print("Running BATCHED (one generate() call for everything)...")
    batched = generate_batched(model, tokenizer, configs, PROMPTS)

    print("\nComparing, row for row...\n")
    all_match = True
    for group_id in configs:
        for i, prompt in enumerate(PROMPTS):
            sep_text = separate[group_id][i]
            bat_text = batched[group_id][i]
            match = sep_text == bat_text
            all_match &= match
            status = "MATCH" if match else "MISMATCH"
            print(f"[{status}] {group_id}, prompt {i}")
            if not match:
                print(f"    separate: {sep_text!r}")
                print(f"    batched:  {bat_text!r}")

    print()
    if all_match:
        print(f"ALL {len(configs) * len(PROMPTS)} generations matched token-for-token. "
              f"Batched heterogeneous steering is confirmed correct on the real model.")
    else:
        print("MISMATCH FOUND -- do not trust batching for a real sweep until this is resolved. "
              "Report this output back before using generate_with_routed_configs for anything real.")
        sys.exit(1)


if __name__ == "__main__":
    main()
