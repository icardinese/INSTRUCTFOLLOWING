"""Sweeps (layer, coefficient) for constant additive steering, generating dev-split outputs under
every grid point. Directions come from contrasting the LAST PROMPT TOKEN (before any generation)
between base and instructed prompts -- classic contrastive activation-steering extraction, distinct
from PSR's response-token pooling (steering/psr/data.py), which contrasts generated tokens instead.

This produces results/<task>/const_steer_directions.pt (one direction per candidate layer) and
sweep_dev.jsonl (every grid point's generations). Picking the WINNING (layer, coefficient) -- the
"give up a little compression for correctness" step -- needs judged output quality, i.e. evals/,
which hasn't been rebuilt in this structure yet. This script deliberately stops at "here are the
candidates," not "here's the answer."
"""
import argparse
import json

import torch

from adapters.registry import get_adapter
from core.model_common import (
    extract_hidden_states,
    generate_response,
    layer_indices_from_fractions,
    load_model,
    num_layers,
)
from steering.const.direction import compute_diff_mean_direction
from steering.const.hooks import make_const_hook
from steering.hooks import steering_hook

COEFF_GRID = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28]


@torch.no_grad()
def compute_directions(model, tokenizer, items: list[dict], layer_indices: list[int]) -> dict[int, torch.Tensor]:
    base_acts = {l: [] for l in layer_indices}
    instr_acts = {l: [] for l in layer_indices}
    for item in items:
        base_h = extract_hidden_states(model, tokenizer, item["base_prompt"], layer_indices=layer_indices)
        instr_h = extract_hidden_states(model, tokenizer, item["terse_prompt"], layer_indices=layer_indices)
        for l in layer_indices:
            base_acts[l].append(base_h[l])
            instr_acts[l].append(instr_h[l])

    directions = {}
    for l in layer_indices:
        try:
            directions[l] = compute_diff_mean_direction(torch.stack(base_acts[l]), torch.stack(instr_acts[l]))
        except ValueError as e:
            print(f"WARNING: skipping layer {l} -- {e}")
    return directions


def main(task: str) -> None:
    adapter = get_adapter(task)
    adapter.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    layer_indices = layer_indices_from_fractions(model)
    print(f"candidate layers: {layer_indices}")

    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))

    directions_path = adapter.RESULTS_DIR / "const_steer_directions.pt"
    if directions_path.exists():
        print(f">>> loading cached directions from {directions_path}")
        directions = torch.load(directions_path)
    else:
        print("computing diff-in-means directions per candidate layer")
        directions = compute_directions(model, tokenizer, dev_items, layer_indices)
        torch.save(directions, directions_path)

    out_path = adapter.RESULTS_DIR / "sweep_dev.jsonl"
    results = []
    done = set()
    if out_path.exists():
        for row in [json.loads(l) for l in out_path.open()]:
            results.append(row)
            done.add((row["id"], row["layer"], row["coeff"]))
        print(f"resuming: {len(done)} (id, layer, coeff) combos already done")

    for layer_idx in layer_indices:
        if layer_idx not in directions:
            continue
        for coeff in COEFF_GRID:
            new_this_config = 0
            for item in dev_items:
                if (item["id"], layer_idx, coeff) in done:
                    continue
                with steering_hook(model, layer_idx, make_const_hook(directions[layer_idx], coeff)):
                    response = generate_response(model, tokenizer, item["base_prompt"])
                results.append({"id": item["id"], "layer": layer_idx, "coeff": coeff, "response": response})
                new_this_config += 1
            if new_this_config:
                print(f"layer={layer_idx} coeff={coeff}: generated {new_this_config} new rows")
                with out_path.open("w") as f:
                    for row in results:
                        f.write(json.dumps(row) + "\n")

    print(f"\nwrote {len(results)} total rows to {out_path}")
    print("NEXT STEP (not this script): judge these for correctness/degeneracy, then pick the "
          "(layer, coefficient) that gives up a little compression for correctness, and write "
          "const_steer_config.json with that choice.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    args = parser.parse_args()
    main(args.task)
